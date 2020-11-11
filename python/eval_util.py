import copy
import logging
import os
import random
from collections import defaultdict
from typing import List, Dict, Callable, Union, Optional

import librenderman as rm
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from audio_features import get_mel_spec, AudioFeatures
from audio_processing_util import ProcessConfig
from audio_rendering import PatchGenerator
from audio_rendering_util import RenderConfig, render_patch
from config import OUT_DIR
from effects import get_effect, PARAM_TO_DESC, DESC_TO_PARAM, PARAM_TO_EFFECT
from models_effect import baseline_cnn_2x
from serum_util import set_preset, setup_serum
from training_rnn import EFFECT_TO_IDX_MAPPING
from training_util import EFFECT_TO_Y_PARAMS
from util import parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

FIXED_EFFECT_SEQ = ['distortion', 'phaser', 'compressor', 'reverb-hall', 'eq']


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


def plot_mel_seq(mel_seq: np.ndarray,
                 effect_seq: np.ndarray,
                 target_effect_names: List[str]) -> None:
    from matplotlib import pyplot as plt
    mel_seq = np.array(mel_seq)
    effect_seq = np.array(effect_seq)
    assert len(mel_seq.shape) == 4
    assert len(effect_seq.shape) == 2
    assert len(effect_seq) == len(mel_seq)
    print(mel_seq.shape)
    print(effect_seq.shape)

    idx_to_effect = {v: k for k, v in EFFECT_TO_IDX_MAPPING.items()}
    target_mel = mel_seq[0, :, :, 0]
    orig_mel = mel_seq[0, :, :, 1]
    plt.imshow(orig_mel, origin='lower')
    plt.xlabel('audio frames')
    plt.ylabel('Mel freq bins')
    plt.title('original audio')
    plt.show()

    effect_names = []
    for mels, effect_tensor in zip(mel_seq[1:], effect_seq[1:]):
        base_mel = mels[:, :, 1]
        effect_name = idx_to_effect[np.argmax(effect_tensor)]
        effect_names.append(effect_name)
        plt.imshow(base_mel, origin='lower')
        plt.xlabel('audio frames')
        plt.ylabel('Mel freq bins')
        plt.title(' + '.join(effect_names))
        plt.show()
    plt.imshow(target_mel, origin='lower')
    plt.xlabel('audio frames')
    plt.ylabel('Mel freq bins')
    plt.title(f'target audio: {" + ".join(target_effect_names)}')
    plt.show()
    plt.imshow(target_mel, origin='lower')
    plt.xlabel('audio frames')
    plt.ylabel('Mel freq bins')
    plt.title(f'target audio: {" + ".join(target_effect_names)}')
    plt.show()


def update_patch(patch: Dict[int, float],
                 rc_effect: Dict[str, Union[str, List[int]]],
                 gran: int,
                 verbose: bool = False) -> None:
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

            if verbose:
                log.info(f'Overriding {effect_name} - {desc} with '
                         f'constant: {const}')
            patch[param] = const_v


def set_default_and_constant_params(
        engine: rm.RenderEngine,
        rc_effects: List[Dict[str, Union[str, List[int]]]],
        orig_rc_effects: List[Dict[str, Union[str, List[int]]]],
        gran: int,
        verbose: bool = False
) -> None:
    orig_rc_effects = {e['name']: e for e in orig_rc_effects}

    for rc_effect in rc_effects:
        effect_name = rc_effect['name']
        effect = get_effect(effect_name)
        patch = effect.default.copy()

        if effect_name in orig_rc_effects:
            update_patch(patch,
                         orig_rc_effects[effect_name],
                         gran=gran,
                         verbose=verbose)

        update_patch(patch, rc_effect, gran=gran)

        if verbose:
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
    for idx in range(batch_size):
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


def render_name_to_rc_effects(
        render_name: str) -> List[Dict[str, Union[str, List[int]]]]:
    render_info = parse_save_name(render_name, is_dir=False)
    rc_effects = {}
    for desc, value in render_info.items():
        if desc == 'name':
            continue
        param = DESC_TO_PARAM[desc]
        effect = PARAM_TO_EFFECT[param]
        effect_name = effect.name
        if effect_name not in rc_effects:
            rc_effects[effect_name] = {'name': effect_name}
        rc_effects[effect_name][desc] = [value]

    rc_effects = rc_effects.values()
    return rc_effects


def get_random_rc_effects(
        rc_effects: List[Dict[str, Union[str, List[int]]]],
        min_n_effects: int = 1,
        max_n_effects: int = -1,
) -> List[Dict[str, Union[str, List[int]]]]:
    n_effects = len(rc_effects)
    assert n_effects > 0
    if max_n_effects == -1:
        max_n_effects = n_effects

    rand_rc_effects = copy.deepcopy(rc_effects)
    n_effects_to_use = np.random.randint(min_n_effects, max_n_effects + 1)
    rand_rc_effects = random.sample(rand_rc_effects,
                                    n_effects_to_use)
    for e in rand_rc_effects:
        for desc, params in e.items():
            if desc == 'name':
                continue
            assert len(params) > 0
            e[desc] = [random.choice(params)]

    return rand_rc_effects


def crunch_eval_data(save_path: str) -> None:
    eval_data = np.load(save_path, allow_pickle=True)
    target_effect_names_all = eval_data['target_effect_names_all'].tolist()
    effect_name_seq_all = eval_data['effect_name_seq_all'].tolist()
    eval_metrics_all = eval_data['eval_metrics_all'].item()

    n_data_points = len(list(eval_metrics_all.items())[0])
    assert all(len(metric) == n_data_points
               for metric in eval_metrics_all.items())

    all_steps_d = defaultdict(lambda: defaultdict(list))
    all_effects_d = defaultdict(lambda: defaultdict(list))
    same_steps_d = defaultdict(lambda: defaultdict(list))

    inits = defaultdict(list)
    all_steps_ends = defaultdict(list)
    all_effects_ends = defaultdict(list)
    same_steps_ends = defaultdict(list)

    for metric_name, metric_all in eval_metrics_all.items():
        # derp = 0

        for target_effect_names, effect_name_seq, metric_vals in zip(
                target_effect_names_all, effect_name_seq_all, metric_all):
            target_effect_names = set(target_effect_names)
            n_target_effects = len(target_effect_names)
            used_effects = set()

            # if derp == 630:
            #     merp = 1
            # derp += 1

            for idx, effect_name in enumerate(effect_name_seq):
                step_idx = idx + 1

                prev_val = metric_vals[idx]
                step_val = metric_vals[step_idx]
                delta = step_val - prev_val

                if idx == 0:
                    inits[metric_name].append(prev_val)
                if step_idx == len(effect_name_seq):
                    all_steps_ends[metric_name].append(step_val)
                if step_idx == n_target_effects:
                    same_steps_ends[metric_name].append(step_val)

                all_steps_d[metric_name][step_idx].append(delta)

                if not all(e in used_effects for e in target_effect_names):
                    all_effects_d[metric_name][step_idx].append(delta)
                prev_effects = set(used_effects)
                used_effects.add(effect_name)
                if not all(e in prev_effects for e in target_effect_names) \
                        and all(e in used_effects for e in target_effect_names):
                    all_effects_ends[metric_name].append(step_val)

                if idx < n_target_effects:
                    same_steps_d[metric_name][step_idx].append(delta)

    for key, init in inits.items():
        all_steps_end = all_steps_ends[key]
        all_effects_end = all_effects_ends[key]
        same_steps_end = same_steps_ends[key]
        mean_init = np.mean(init)

        mean_all_steps_end = np.mean(all_steps_end)
        mean_all_effects_end = np.mean(all_effects_end)
        mean_same_steps_end = np.mean(same_steps_end)
        log.info(f'{key} mean init error = {mean_init:.4f}')

        log.info(f'{key} mean all_steps_end error = {mean_all_steps_end:.4f}')
        log.info(f'{key} mean all_effects_end error = {mean_all_effects_end:.4f}')
        log.info(f'{key} mean same_steps_end error = {mean_same_steps_end:.4f}')

        log.info(f'{key} mean all_steps_end error d = {mean_all_steps_end - mean_init:.4f}')
        log.info(f'{key} mean mean_all_effects_end error d = {mean_all_effects_end - mean_init:.4f}')
        log.info(f'{key} mean mean_same_steps_end error d = {mean_same_steps_end - mean_init:.4f}')
        log.info('')

        all_steps_c = all_steps_d[key]
        all_effects_c = all_effects_d[key]
        same_steps_c = same_steps_d[key]
        for step_idx, all_steps_s in all_steps_c.items():
            all_effects_s = all_effects_c[step_idx]
            same_steps_s = same_steps_c[step_idx]
            log.info(f'step_idx = {step_idx}')
            log.info(f'all_steps n={len(all_steps_s)}, mean d = {np.mean(all_steps_s):.4f}')
            log.info(f'all_effects n={len(all_effects_s)}, mean d = {np.mean(all_effects_s):.4f}')
            log.info(f'same_steps n={len(same_steps_s)}, mean d = {np.mean(same_steps_s):.4f}')
            log.info('')
        log.info('')

        # print(max(init))
        # print(init.index(max(init)))


def effect_cnn_audio_step(
        preset_path: str,
        step_rc_effects: List[Dict[str, Union[str, List[int]]]],
        rc: RenderConfig,
        pc: ProcessConfig,
        engine: Optional[rm.RenderEngine] = None,
        patch: Optional[Dict[int, float]] = None,
        render_save_dir: Optional[str] = OUT_DIR,
        render_save_name: Optional[str] = None
) -> (rm.RenderEngine, np.ndarray, AudioFeatures):
    if engine is None:
        engine = setup_serum(preset_path, sr=rc.sr, render_once=True)
    if patch is None:
        patch = {}

    set_default_and_constant_params(engine,
                                    step_rc_effects,
                                    rc.effects,
                                    rc.gran)
    audio = render_patch(engine,
                         patch,
                         rc,
                         render_save_dir,
                         render_save_name)
    audio_features = get_mel_spec(audio=audio,
                                  sr=pc.sr,
                                  n_fft=pc.n_fft,
                                  hop_length=pc.hop_length,
                                  n_mels=pc.n_mels,
                                  max_n_of_frames=pc.max_n_of_frames,
                                  norm_audio=pc.norm_audio,
                                  norm_mel=pc.norm_mel,
                                  fmin=pc.fmin,
                                  fmax=pc.fmax,
                                  db_ref=pc.db_ref,
                                  top_db=pc.top_db,
                                  n_mfcc=pc.n_mfcc,
                                  calc_cent=pc.calc_cent,
                                  calc_bw=pc.calc_bw,
                                  calc_flat=pc.calc_flat)

    return engine, audio, audio_features


def create_effect_cnn_x(target_af: AudioFeatures,
                        base_af: AudioFeatures) -> List[np.ndarray]:
    mel_x = np.stack([target_af.mel, base_af.mel], axis=-1)
    mel_x = np.expand_dims(mel_x, axis=0)
    mfcc_x = np.stack([target_af.mfcc, base_af.mfcc], axis=-1)
    mfcc_x = np.expand_dims(mfcc_x, axis=0)
    return [mel_x, mfcc_x]
