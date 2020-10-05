import logging
import os
from typing import List, Dict, Callable, Union

import librenderman as rm
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tqdm import tqdm

from audio_processing import ProcessConfig, get_mel_spec
from audio_rendering import RenderConfig, PatchGenerator, render_patch
from config import OUT_DIR, DATASETS_DIR, MODELS_DIR, CONFIGS_DIR
from effects import get_effect, PARAM_TO_DESC, DESC_TO_PARAM
from models import baseline_cnn_2x, baseline_effect_rnn, baseline_cnn
from serum_util import setup_serum, set_preset
from training_rnn import EFFECT_TO_IDX_MAPPING, get_x_ids, RNNDataGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


EFFECT_TO_Y_PARAMS = {
    'compressor': {270, 271, 272},
    'distortion': {97, 99},
    'eq': {88, 90, 92, 94},
    'flanger': {105, 106, 107},
    'phaser': {111, 112, 113, 114},
}


def load_effect_cnns(models_dir: str,
                     model_prefix: str,
                     architecture: Callable = baseline_cnn_2x,
                     channel_mode: int = 1) -> Dict[str, Model]:
    cnns = {}

    for effect_name in EFFECT_TO_IDX_MAPPING:
        model_name = f'{model_prefix}__{effect_name}__{architecture.__name__}' \
                     f'__cm_{channel_mode}__best.h5'
        model_path = os.path.join(models_dir, model_name)
        log.info(f'Loading {model_name}')
        effect_cnn = load_model(model_path)
        cnns[effect_name] = effect_cnn

    return cnns


def update_patch(patch: Dict[int, float],
                 rc_effect: Dict[str, Union[str, List[int]]],
                 gran: int) -> None:
    effect_name = rc_effect['name']
    effect = get_effect(effect_name)

    for desc, values in rc_effect.items():
        if desc == 'name':
            continue
        if len(values) == 1:
            const = values[0]
            param = DESC_TO_PARAM[desc]
            assert param in effect.default
            if param in effect.continuous:
                const_v = float(const / gran)
            elif param in effect.binary:
                assert const == 0 or const == 1
                const_v = float(const)
            else:
                n_categories = effect.categorical[param]
                const_v = float(const / (n_categories - 1))
            log.info(f'Overriding {effect_name} - {desc} with '
                     f'constant: {const}')
            patch[param] = const_v


def set_default_and_constant_params(
        engine: rm.RenderEngine,
        rc_effects: List[Dict[str, Union[str, List[int]]]],
        orig_rc_effects: List[Dict[str, Union[str, List[int]]]],
        gran: int
) -> None:
    orig_rc_effects = {e['name']: e for e in orig_rc_effects}

    for rc_effect in rc_effects:
        effect_name = rc_effect['name']
        effect = get_effect(effect_name)
        patch = effect.default.copy()

        if effect_name in orig_rc_effects:
            update_patch(patch, orig_rc_effects[effect_name], gran=gran)

        update_patch(patch, rc_effect, gran=gran)

        log.info(f'Setting {effect_name} default and constant params.')
        set_preset(engine, patch)


def get_patch_from_effect_cnn(effect_name: str,
                              pred: List[np.ndarray],
                              gran: int,
                              batch_size: int = 1) -> List[Dict[int, float]]:
    assert isinstance(pred, list)
    effect = get_effect(effect_name)
    y_params = sorted(list(EFFECT_TO_Y_PARAMS[effect_name]))
    bin_params = [p for p in y_params if p in effect.binary]
    cate_params = [p for p in y_params if p in effect.categorical]
    cont_params = [p for p in y_params if p in effect.continuous]

    pred_y_s = pred.copy()
    pred_bin = None
    pred_cate_s = None
    pred_cont = None

    if bin_params:
        pred_bin = pred_y_s[0]
        pred_y_s = pred_y_s[1:]
        pred_bin = np.around(pred_bin).astype(np.int32)

    if cont_params:
        pred_cont = pred_y_s[-1]
        pred_y_s = pred_y_s[:-1]
        pred_cont = np.around(pred_cont * gran).astype(np.int32)

    if cate_params:
        pred_cate_s = [np.argmax(_, axis=-1) for _ in pred_y_s]

    rc_effect = {'name': effect_name}

    patches = []
    for idx in tqdm(range(batch_size)):
        if pred_bin is not None:
            pred_bin_row = pred_bin[idx]
            for bin_param, pred_bin_v in zip(bin_params, pred_bin_row):
                desc = PARAM_TO_DESC[bin_param]
                rc_effect[desc] = [pred_bin_v]

        if pred_cate_s is not None:
            for cate_param, pred_cate in zip(cate_params, pred_cate_s):
                pred_cate_v = pred_cate[idx]
                desc = PARAM_TO_DESC[cate_param]
                rc_effect[desc] = [pred_cate_v]

        if pred_cont is not None:
            pred_cont_row = pred_cont[idx]
            for cont_param, pred_cont_v in zip(cont_params, pred_cont_row):
                desc = PARAM_TO_DESC[cont_param]
                rc_effect[desc] = [pred_cont_v]

        rc_effects = [rc_effect]
        pg = PatchGenerator(gran, rc_effects)
        assert pg.n_combos == 1

        _, patch = list(pg.generate_all_combos())[0]
        patches.append(patch)

    return patches


def get_next_effect_name(rnn_pred: np.ndarray,
                         effect_idx_to_name: Dict[int, str],
                         effects_can_repeat: bool,
                         effect_name_seq: List[str]) -> str:
    assert len(rnn_pred.shape) == 1

    if effects_can_repeat:
        next_effect_idx = np.argmax(rnn_pred)
        next_effect_name = effect_idx_to_name[next_effect_idx]
        return next_effect_name

    used_effects = set(effect_name_seq)
    n_effects = len(effect_idx_to_name)
    assert len(used_effects) < n_effects

    next_effect_name = None
    highest_prob = 0.0
    for idx, prob in enumerate(rnn_pred):
        effect_name = effect_idx_to_name[idx]
        if effect_name not in used_effects and prob > highest_prob:
            next_effect_name = effect_name
            highest_prob = prob

    return next_effect_name


def ensemble(init_rc_effects: List[Dict[str, Union[str, List[int]]]],
             target_rc_effects: List[Dict[str, Union[str, List[int]]]],
             rc: RenderConfig,
             pc: ProcessConfig,
             renders_save_dir: str,
             rnn: Model,
             cnns: Dict[str, Model],
             effect_name_to_idx: Dict[str, int] = EFFECT_TO_IDX_MAPPING,
             effects_can_repeat: bool = False,
             max_steps: int = 8):
    rc.use_hashes = False
    log.info(f'Using preset: {rc.preset}')

    effect_idx_to_name = {v: k for k, v in effect_name_to_idx.items()}
    n_effects = len(effect_name_to_idx)
    if not effects_can_repeat:
        max_steps = n_effects

    engine = setup_serum(rc.preset, sr=rc.sr, render_once=True)
    set_default_and_constant_params(engine,
                                    init_rc_effects,
                                    rc.effects,
                                    gran=rc.gran)

    init_effect_names = sorted([e['name'] for e in init_rc_effects])
    render_name = f'00_init__{"_".join(init_effect_names)}.wav'
    init_audio = render_patch(engine, {}, rc, renders_save_dir, render_name)
    init_mel = get_mel_spec(init_audio,
                            pc.sr,
                            hop_length=pc.hop_length,
                            n_mels=pc.n_mels,
                            n_fft=pc.n_fft,
                            max_n_of_frames=pc.max_n_of_frames,
                            normalize_audio=pc.normalize_audio,
                            normalize_mel=pc.normalize_mel)

    target_engine = setup_serum(rc.preset, sr=rc.sr, render_once=True)
    set_default_and_constant_params(target_engine,
                                    target_rc_effects,
                                    rc.effects,
                                    gran=rc.gran)
    target_effect_names = sorted([e['name'] for e in target_rc_effects])
    render_name = f'target__{"_".join(target_effect_names)}.wav'
    target_audio = render_patch(target_engine,
                                {},
                                rc,
                                renders_save_dir,
                                render_name)
    target_mel = get_mel_spec(target_audio,
                              pc.sr,
                              hop_length=pc.hop_length,
                              n_mels=pc.n_mels,
                              n_fft=pc.n_fft,
                              max_n_of_frames=pc.max_n_of_frames,
                              normalize_audio=pc.normalize_audio,
                              normalize_mel=pc.normalize_mel)

    init_mse = mean_squared_error(init_mel, target_mel)
    init_mae = mean_absolute_error(init_mel, target_mel)
    log.info(f'init_effect_names = {init_effect_names}')
    log.info(f'target_effect_names = {target_effect_names}')
    log.info(f'init_mse = {init_mse:.4f}')
    log.info(f'init_mae = {init_mae:.4f}')

    mel_seq = [np.stack([target_mel, init_mel], axis=-1)]

    init_effect = np.zeros((n_effects + 1,), dtype=np.float32)
    init_effect[-1] = 1.0

    effect_name_seq = []
    effect_seq = [init_effect]
    mses = [init_mse]
    maes = [init_mae]
    patch_seq = []

    for step_idx in range(max_steps):
        rnn_x = (np.expand_dims(np.array(mel_seq, dtype=np.float32), axis=0),
                 np.expand_dims(np.array(effect_seq, dtype=np.float32), axis=0))
        rnn_pred = rnn.predict(rnn_x, batch_size=1)[0]
        log.info(f'rnn_pred = {rnn_pred}')

        next_effect_name = get_next_effect_name(rnn_pred,
                                                effect_idx_to_name,
                                                effects_can_repeat,
                                                effect_name_seq)
        log.info(f'next_effect_name = {next_effect_name}')
        effect_name_seq.append(next_effect_name)
        next_rc_effects = [{'name': next_effect_name}]

        cnn = cnns[next_effect_name]
        cnn_x = np.expand_dims(mel_seq[-1], axis=0)
        cnn_pred = cnn.predict(cnn_x, batch_size=1)
        if not isinstance(cnn_pred, list):
            cnn_pred = [cnn_pred]

        set_default_and_constant_params(engine,
                                        next_rc_effects,
                                        rc.effects,
                                        gran=rc.gran)
        effect_names = sorted(list(set(effect_name_seq)))
        render_name = f'{step_idx + 1:02d}__{"_".join(effect_names)}.wav'
        patches = get_patch_from_effect_cnn(next_effect_name,
                                            cnn_pred,
                                            rc.gran,
                                            batch_size=1)
        patch = patches[0]
        log.info(f'patch = {patch}')
        patch_seq.append(patch)

        next_audio = render_patch(engine,
                                  patch,
                                  rc,
                                  renders_save_dir,
                                  render_name)
        next_mel = get_mel_spec(next_audio,
                                pc.sr,
                                hop_length=pc.hop_length,
                                n_mels=pc.n_mels,
                                n_fft=pc.n_fft,
                                max_n_of_frames=pc.max_n_of_frames,
                                normalize_audio=pc.normalize_audio,
                                normalize_mel=pc.normalize_mel)
        next_mse = mean_squared_error(next_mel, target_mel)
        next_mae = mean_absolute_error(next_mel, target_mel)
        log.info(f'next_mse = {next_mse:.4f}')
        log.info(f'next_mae = {next_mae:.4f}')
        mses.append(next_mse)
        maes.append(next_mae)

        mel_seq.append(np.stack([next_mel, init_mel], axis=-1))
        next_effect_idx = effect_name_to_idx[next_effect_name]
        next_effect = np.zeros((n_effects + 1,), dtype=np.float32)
        assert next_effect_idx != n_effects
        next_effect[next_effect_idx] = 1.0
        effect_seq.append(next_effect)

    print(effect_name_seq)
    print(mses)
    print(maes)
    print(patch_seq)


if __name__ == '__main__':
    render_config_path = os.path.join(CONFIGS_DIR, 'rendering/seq_5_train.yaml')
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)
    rc = RenderConfig(**render_config)

    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)

    init_rc_effects = []
    target_rc_effects = [
        {'name': 'compressor', 'CompMB L': [60], 'CompMB M': [60], 'CompMB H': [70]},
        {'name': 'distortion', 'Dist_Drv': [74], 'Dist_Mode': [5]},
        {'name': 'eq', 'EQ FrqL': [50], 'EQ Q L': [100], 'EQ VolL': [0], 'EQ TypL': [0]},
        {'name': 'flanger', 'Flg_Rate': [50], 'Flg_Dep': [40], 'Flg_Feed': [60]},
        {'name': 'phaser', 'Phs_Rate': [75]},
    ]
    renders_save_dir = OUT_DIR

    rnn_model_name = 'basic_shapes__rnn__baseline_cnn__best.h5'
    rnn = load_model(os.path.join(MODELS_DIR, rnn_model_name))
    cnns = load_effect_cnns(MODELS_DIR, 'basic_shapes_exclude_all')

    ensemble(init_rc_effects,
             target_rc_effects,
             rc,
             pc,
             renders_save_dir,
             rnn,
             cnns,
             effects_can_repeat=False,
             max_steps=8)
    exit()

    in_x = 128
    in_y = 88
    n_channels = 2
    n_effects = len(EFFECT_TO_IDX_MAPPING)

    cnn_architecture = baseline_cnn
    # cnn_architecture = baseline_cnn_2x

    batch_size = 32
    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    model_name = f'basic_shapes__rnn__{cnn_architecture.__name__}__best.h5'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__rnn')
    data_dir = os.path.join(datasets_dir, f'basic_shapes__rnn')

    _, _, test_x_ids = get_x_ids(data_dir)
    log.info(f'test_x_ids length = {len(test_x_ids)}')

    test_gen = RNNDataGenerator(test_x_ids,
                                in_x,
                                in_y,
                                n_effects,
                                effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                                batch_size=batch_size)

    model = baseline_effect_rnn(in_x,
                                in_y,
                                n_channels,
                                n_effects,
                                cnn_architecture=cnn_architecture)
    # model.load_weights(os.path.join(MODELS_DIR, model_name))
    model.load_weights(os.path.join(OUT_DIR, model_name))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics='acc')
    model.summary()

    eval_results = model.evaluate(test_gen,
                                  use_multiprocessing=use_multiprocessing,
                                  workers=workers,
                                  return_dict=True,
                                  verbose=1)
    log.info(f'{eval_results}')
