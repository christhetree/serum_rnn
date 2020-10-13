import logging
import os
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from audio_processing_util import ProcessConfig, get_mel_spec
from audio_rendering import RenderConfig, render_patch
from config import OUT_DIR, MODELS_DIR, CONFIGS_DIR, DATASETS_DIR
from effects import DESC_TO_PARAM
from eval_util import get_patch_from_effect_cnn, \
    set_default_and_constant_params, get_next_effect_name, \
    render_name_to_rc_effects, get_random_rc_effects, load_effect_cnns
from metrics import calc_mfcc_metric, calc_lsd
from serum_util import setup_serum
from training_rnn import EFFECT_TO_IDX_MAPPING

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def ensemble(init_rc_effects: List[Dict[str, Union[str, List[int]]]],
             target_rc_effects: List[Dict[str, Union[str, List[int]]]],
             preset_path: str,
             rc: RenderConfig,
             pc: ProcessConfig,
             renders_save_dir: str,
             rnn: Model,
             cnns: Dict[str, Model],
             effect_name_to_idx: Dict[str, int] = EFFECT_TO_IDX_MAPPING,
             effects_can_repeat: bool = False,
             max_steps: int = 8):
    rc.use_hashes = False
    rc.preset = None
    log.info(f'Using preset: {preset_path}')

    effect_idx_to_name = {v: k for k, v in effect_name_to_idx.items()}
    n_effects = len(effect_name_to_idx)
    if not effects_can_repeat:
        max_steps = n_effects

    engine = setup_serum(preset_path, sr=rc.sr, render_once=True, instance=1)
    set_default_and_constant_params(engine,
                                    init_rc_effects,
                                    rc.effects,
                                    gran=rc.gran)

    init_effect_names = sorted([e['name'] for e in init_rc_effects])
    # render_name = f'0_init__{"_".join(init_effect_names)}.wav'
    render_name = f'00_init.wav'
    init_audio = render_patch(engine, {}, rc, renders_save_dir, render_name)
    init_audio = render_patch(engine, {}, rc, renders_save_dir, render_name)
    init_mel = get_mel_spec(init_audio,
                            pc.sr,
                            hop_length=pc.hop_length,
                            n_mels=pc.n_mels,
                            n_fft=pc.n_fft,
                            max_n_of_frames=pc.max_n_of_frames,
                            normalize_audio=pc.normalize_audio,
                            normalize_mel=pc.normalize_mel)

    target_engine = setup_serum(preset_path, sr=rc.sr, render_once=True, instance=2)
    set_default_and_constant_params(target_engine,
                                    target_rc_effects,
                                    rc.effects,
                                    gran=rc.gran)
    target_effect_names = sorted([e['name'] for e in target_rc_effects])
    render_name = f'00_target__{"_".join(target_effect_names)}.wav'
    target_audio = render_patch(target_engine,
                                {},
                                rc,
                                renders_save_dir,
                                render_name)
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
    init_mfcc_dist = calc_mfcc_metric(target_mel,
                                      init_mel,
                                      n_mfcc=20,
                                      sr=pc.sr,
                                      normalized_mel=pc.normalize_mel)
    init_lsd = calc_lsd(target_mel,
                        init_mel,
                        normalized_mel=pc.normalize_mel)
    log.info(f'init_effect_names = {init_effect_names}')
    log.info(f'target_effect_names = {target_effect_names}')
    log.info(f'init_mse = {init_mse:.6f}')
    log.info(f'init_mae = {init_mae:.6f}')
    log.info(f'init_mfcc_dist = {init_mfcc_dist:.6f}')
    log.info(f'init_lsd = {init_lsd:.6f}')

    mel_seq = [np.stack([target_mel, init_mel], axis=-1)]

    init_effect = np.zeros((n_effects + 1,), dtype=np.float32)
    init_effect[-1] = 1.0

    effect_name_seq = []
    effect_seq = [init_effect]
    mses = [init_mse]
    maes = [init_mae]
    mfcc_dists = [init_mfcc_dist]
    lsds = [init_lsd]
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
        next_mfcc_dist = calc_mfcc_metric(target_mel,
                                          next_mel,
                                          n_mfcc=20,
                                          sr=pc.sr,
                                          normalized_mel=pc.normalize_mel)
        next_lsd = calc_lsd(target_mel,
                            next_mel,
                            normalized_mel=pc.normalize_mel)

        log.info(f'next_mse = {next_mse:.6f}')
        log.info(f'next_mae = {next_mae:.6f}')
        log.info(f'next_mfcc_dist = {next_mfcc_dist:.6f}')
        log.info(f'next_lsd = {next_lsd:.6f}')
        mses.append(next_mse)
        maes.append(next_mae)
        mfcc_dists.append(next_mfcc_dist)
        lsds.append(next_lsd)

        mel_seq.append(np.stack([target_mel, next_mel], axis=-1))
        next_effect_idx = effect_name_to_idx[next_effect_name]
        next_effect = np.zeros((n_effects + 1,), dtype=np.float32)
        assert next_effect_idx != n_effects
        next_effect[next_effect_idx] = 1.0
        effect_seq.append(next_effect)

    # plot_mel_seq(mel_seq, effect_seq, target_effect_names)
    log.info('')
    log.info('')
    log.info('Results:')
    log.info('')
    log.info(f'target effects  = {target_effect_names}')
    log.info(f'effect_name_seq = {effect_name_seq}')
    log.info(f'mses = {" -> ".join(f"{mse:.4f}" for mse in mses)}')
    log.info(f'maes = {" -> ".join(f"{mae:.4f}" for mae in maes)}')
    log.info(f'mfcc_dists = {" -> ".join(f"{dist:.4f}" for dist in mfcc_dists)}')
    log.info(f'lsds = {" -> ".join(f"{lsd:.4f}" for lsd in lsds)}')
    log.info('')
    log.info('target effect values')

    for e in target_rc_effects:
        log.info(f'{sorted(e.items())}')
        for desc in sorted(e):
            if desc == 'name':
                continue
            param = DESC_TO_PARAM[desc]
            target_v = target_engine.get_parameter(param)
            actual_v = engine.get_parameter(param)
            log.info(f'{desc.ljust(12)} = actual {actual_v:.4f} | '
                     f'{target_v:.4f} target')

    return target_effect_names, effect_name_seq, mses, maes, mfcc_dists, lsds


def prepare_ensemble_render_names(presets_cat: str):
    rows = []
    for effect_name in EFFECT_TO_IDX_MAPPING:
        eval_data = np.load(os.path.join(DATASETS_DIR, f'seq_5_v3__{presets_cat}__{effect_name}__baseline_cnn_2x__cm_1__eval_spec_data.npz'))
        render_names = eval_data['render_names']
        presets = eval_data['presets']
        for row in zip(render_names, presets):
            rows.append(row)

    log.info(f'rows length = {len(rows)}')
    np.random.shuffle(rows)

    preset_count = defaultdict(int)
    rows_by_seq_length = defaultdict(list)
    for render_name, preset in rows:
        rc_effects = render_name_to_rc_effects(render_name)
        n_effects = len(rc_effects)
        preset_count[preset] += 1
        rows_by_seq_length[n_effects].append((render_name, preset))

    print(preset_count)
    final_rows = []
    for k, v in sorted(rows_by_seq_length.items()):
        print(f'{k} = {len(v)}')
        final_rows.extend(v[:200])

    print(f'{len(final_rows)}')
    render_names, presets = zip(*final_rows)
    preset_count = defaultdict(int)
    for preset in presets:
        preset_count[preset] += 1
    print(preset_count)
    np.savez(os.path.join(DATASETS_DIR, f'seq_5_v3__{presets_cat}__ensemble_eval_in.npz'),
             render_names=render_names,
             presets=presets)


if __name__ == '__main__':
    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    # save_path = os.path.join(DATASETS_DIR, f'seq_5_v3__{presets_cat}__ensemble_eval_data.npz')
    # crunch_eval_data(save_path)
    # exit()

    # prepare_ensemble_render_names(presets_cat)
    # exit()

    render_config_path = os.path.join(CONFIGS_DIR,
                                      'rendering/seq_5_v3_train.yaml')
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)
    rc = RenderConfig(**render_config)

    test_rc_path = os.path.join(CONFIGS_DIR, 'rendering/seq_5_v3_test.yaml')
    with open(test_rc_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)
    test_rc = RenderConfig(**render_config)
    test_rc_effects = test_rc.effects
    rand_rc_effects = get_random_rc_effects(test_rc_effects, 2, 5)

    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)

    init_rc_effects = []
    # target_rc_effects = [
    #     {'name': 'compressor', 'CompMB L': [60], 'CompMB M': [60], 'CompMB H': [70]},
    #     {'name': 'distortion', 'Dist_Drv': [74], 'Dist_Mode': [5]},
    #     {'name': 'eq', 'EQ FrqL': [50], 'EQ Q L': [100], 'EQ VolL': [0], 'EQ TypL': [0]},
    #     {'name': 'flanger', 'Flg_Rate': [50], 'Flg_Dep': [40], 'Flg_Feed': [60]},
    #     {'name': 'phaser', 'Phs_Rate': [75]},
    # ]

    # target_render_name = 'distortion_phaser__Dist_Drv_082__Dist_Mode_07__Phs_Dpth_086__Phs_Frq_057__Phs_Feed_078'
    # target_render_name = 'distortion_compressor_eq_reverb-hall.wav'

    renders_save_dir = OUT_DIR

    rnn_model_name = f'seq_5_v3__{presets_cat}__rnn__baseline_cnn__best.h5'
    # rnn_model_name = 'random_baseline_effect_rnn.h5'
    rnn = load_model(os.path.join(MODELS_DIR, rnn_model_name))
    cnns = load_effect_cnns(MODELS_DIR, f'seq_5_v3__{presets_cat}')

    # eval_in_data = np.load(os.path.join(DATASETS_DIR, f'seq_5_v3__{presets_cat}__ensemble_eval_in.npz'))
    # render_names = eval_in_data['render_names']
    # presets = eval_in_data['presets']
    # assert len(render_names) == len(presets)
    #
    # save_path = os.path.join(DATASETS_DIR, f'seq_5_v3__{presets_cat}__ensemble_eval_data.npz')
    # target_effect_names_all = []
    # effect_name_seq_all = []
    # mses_all = []
    # maes_all = []
    # mfcc_dists_all = []
    # lsds_all = []
    #
    # if os.path.exists(save_path):
    #     eval_data = np.load(save_path, allow_pickle=True)
    #     target_effect_names_all = eval_data['target_effect_names_all'].tolist()
    #     effect_name_seq_all = eval_data['effect_name_seq_all'].tolist()
    #     mses_all = eval_data['mses_all'].tolist()
    #     maes_all = eval_data['maes_all'].tolist()
    #     mfcc_dists_all = eval_data['mfcc_dists_all'].tolist()
    #     lsds_all = eval_data['lsds_all'].tolist()

    # n_completed = len(target_effect_names_all)
    # log.info(f'n_completed = {n_completed}')
    # r_p = list(zip(render_names, presets))
    # np.random.shuffle(r_p)

    # for idx, (render_name, preset) in enumerate(r_p):
    for idx, (target_rc_effects) in enumerate([rand_rc_effects]):
        # if idx < n_completed:
        #     continue

        # target_rc_effects = render_name_to_rc_effects(render_name)
        preset_path = test_rc.preset
        # preset_path = os.path.join(PRESETS_DIR, f'{preset}.fxp')
        # preset_path = os.path.join(PRESETS_DIR, f'subset/{preset}.fxp')
        assert os.path.exists(preset_path)
        log.info(f'preset_path = {preset_path}')

        target_effect_names, \
        effect_name_seq, \
        mses, \
        maes, \
        mfcc_dists, \
        lsds = ensemble(init_rc_effects,
                        target_rc_effects,
                        preset_path,
                        rc,
                        pc,
                        renders_save_dir,
                        rnn,
                        cnns,
                        effects_can_repeat=False)

        break
        # target_effect_names_all.append(target_effect_names)
        # effect_name_seq_all.append(effect_name_seq)
        # mses_all.append(mses)
        # maes_all.append(maes)
        # mfcc_dists_all.append(mfcc_dists)
        # lsds_all.append(lsds)
        #
        # np.savez(save_path,
        #          target_effect_names_all=target_effect_names_all,
        #          effect_name_seq_all=effect_name_seq_all,
        #          mses_all=mses_all,
        #          maes_all=maes_all,
        #          mfcc_dists_all=mfcc_dists_all,
        #          lsds_all=lsds_all)
