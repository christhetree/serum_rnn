import logging
import os
from collections import defaultdict
from typing import List, Dict, Union, Callable, DefaultDict, Tuple

import numpy as np
import yaml
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from audio_processing_util import ProcessConfig
from audio_rendering import RenderConfig
from config import OUT_DIR, MODELS_DIR, CONFIGS_DIR, DATA_DIR, PRESETS_DIR
from effects import DESC_TO_PARAM
from eval_util import get_patch_from_effect_cnn, render_name_to_rc_effects, \
    load_effect_cnns, effect_cnn_audio_step, create_effect_cnn_x, \
    get_random_rc_effects
from metrics import mse, mae, mfcc_dist, lsd, pcc, mssmae
from models_next_effect import next_effect_rnn, next_effect_seq_only_rnn, \
    all_effects_cnn
from next_effect_wrappers import NextEffectWrapper, NextEffectRNNWrapper, \
    NextEffectSeqOnlyRNNWrapper, AllEffectsCNNWrapper, OracleWrapper, \
    RandomWrapper
from training_rnn import EFFECT_TO_IDX_MAPPING

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def prepare_ensemble_render_names(eval_in_dir: str,
                                  presets_cat: str) -> None:
    rows = []
    for effect_name in EFFECT_TO_IDX_MAPPING:
        data_name = f'seq_5_v3__mfcc_30__{presets_cat}__{effect_name}' \
                    f'__baseline_cnn_2x__cm_1__eval_in_data.npz'
        eval_data = np.load(os.path.join(eval_in_dir, data_name))
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

    log.info(f'preset_count = {preset_count}')
    log.info(f'rows_by_seq_length')
    final_rows = []
    for k, v in sorted(rows_by_seq_length.items()):
        log.info(f'{k} = {len(v)}')
        final_rows.extend(v[:200])

    np.random.shuffle(final_rows)
    log.info(f'len final_rows = {len(final_rows)}')
    render_names, presets = zip(*final_rows)
    preset_count = defaultdict(int)
    for preset in presets:
        preset_count[preset] += 1
    log.info(f'preset_count = {preset_count}')
    np.savez(os.path.join(eval_in_dir, f'seq_5_v3__mfcc_30__{presets_cat}'
                                       f'__ensemble__eval_in_data.npz'),
             render_names=render_names,
             presets=presets)


def ensemble(init_rc_effects: List[Dict[str, Union[str, List[int]]]],
             target_rc_effects: List[Dict[str, Union[str, List[int]]]],
             preset_path: str,
             rc: RenderConfig,
             pc: ProcessConfig,
             renders_save_dir: str,
             renders_prefix: Union[str, int],
             next_effect_wrapper: NextEffectWrapper,
             effect_models: Dict[str, Model],
             mel_metrics: List[Callable[[np.ndarray, np.ndarray], float]],
             audio_metrics: List[Callable[[np.ndarray, np.ndarray],
                                          List[Tuple[str, float]]]],
             save_renders: bool = True,
             effects_can_repeat: bool = False,
             max_steps: int = 8,
             seq_effects: bool = True) -> (List[str],
                                           List[str],
                                           DefaultDict[str, List[float]]):
    log.info(f'Using preset: {preset_path}')

    n_effects = next_effect_wrapper.n_effects
    if not effects_can_repeat:
        max_steps = n_effects

    metric_steps = defaultdict(list)

    target_effect_names = sorted([e['name'] for e in target_rc_effects])
    render_save_name = None
    if save_renders:
        render_save_name = f'{renders_prefix}__1_tar__00__target' \
                           f'__{"_".join(target_effect_names)}.wav'

    double_render = True
    if set(target_effect_names) == {'reverb-hall'}:
        log.info('Double render for target and init audio turned off.')
        double_render = False

    target_engine, target_audio, target_af = effect_cnn_audio_step(
        preset_path,
        target_rc_effects,
        rc,
        pc,
        render_save_dir=renders_save_dir,
        render_save_name=render_save_name,
        double_render=double_render
    )

    if save_renders:
        init_effect_names = sorted([e['name'] for e in init_rc_effects])
        render_save_name = f'{renders_prefix}__2_mid__00__init'  \
                           f'__{"_".join(init_effect_names)}.wav'

    engine, dry_audio, dry_af = effect_cnn_audio_step(
        preset_path,
        init_rc_effects,
        rc,
        pc,
        render_save_dir=renders_save_dir,
        render_save_name=render_save_name,
        double_render=double_render
    )

    for metric in mel_metrics:
        dry_v = metric(dry_af.mel, target_af.mel)
        metric_steps[metric.__name__].append(dry_v)
    for metric in audio_metrics:
        metric_results = metric(dry_audio, target_audio)
        for metric_name, dry_v in metric_results:
            metric_steps[metric_name].append(dry_v)

    af_seq = [dry_af]
    effect_name_seq = []

    for step_idx in range(max_steps):
        next_effect_name = next_effect_wrapper.get_next_effect_name(
            target_af,
            af_seq,
            effect_name_seq,
            effects_can_repeat,
            target_effect_names
        )
        log.info(f'next_effect_name = {next_effect_name}')
        effect_name_seq.append(next_effect_name)

        effect_model = effect_models[next_effect_name]

        base_af = af_seq[-1]
        cnn_x = create_effect_cnn_x(target_af, base_af)
        cnn_pred = effect_model.predict(cnn_x, batch_size=1)
        if not isinstance(cnn_pred, list):
            cnn_pred = [cnn_pred]
        cnn_pred[-1] = np.clip(cnn_pred[-1], a_min=0.0, a_max=1.0)

        patches = get_patch_from_effect_cnn(next_effect_name,
                                            cnn_pred,
                                            rc.gran,
                                            batch_size=1)
        patch = patches[0]
        log.info(f'pred patch = {patch}')
        step_rc_effects = [{'name': next_effect_name}]

        if save_renders:
            render_save_name = f'{renders_prefix}__2_mid__{step_idx + 1:02d}' \
                               f'__{"_".join(effect_name_seq)}.wav'
            if len(effect_name_seq) == len(target_effect_names):
                render_save_name = f'{renders_prefix}' \
                                   f'__1_end__{step_idx + 1:02d}' \
                                   f'__{"_".join(effect_name_seq)}.wav'

            if len(effect_name_seq) > len(target_effect_names):
                render_save_name = None

        double_render = True
        if next_effect_name == 'reverb-hall':
            log.info('Double render for wet audio turned off.')
            double_render = False

        _, wet_audio, wet_af = effect_cnn_audio_step(
            preset_path,
            step_rc_effects,
            rc,
            pc,
            engine=engine,
            patch=patch,
            render_save_dir=renders_save_dir,
            render_save_name=render_save_name,
            double_render=double_render
        )

        for metric in mel_metrics:
            wet_v = metric(wet_af.mel, target_af.mel)
            metric_steps[metric.__name__].append(wet_v)
        for metric in audio_metrics:
            metric_results = metric(wet_audio, target_audio)
            for metric_name, wet_v in metric_results:
                metric_steps[metric_name].append(wet_v)

        if seq_effects:
            af_seq.append(wet_af)
        else:
            af_seq.append(dry_af)

    log.info('')
    log.info('Results:')
    log.info('')
    log.info(f'target effects  = {target_effect_names}')
    log.info(f'effect_name_seq = {effect_name_seq}')
    log.info('')

    for metric_name, values in metric_steps.items():
        log.info(f'{metric_name:<9} = '
                 f'{" -> ".join(f"{v:>8.4f}" for v in values)}')

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
            log.info(f'{desc:<12} = actual {actual_v:.4f} | '
                     f'{target_v:.4f} target')
    log.info('')
    log.info('')

    return target_effect_names, effect_name_seq, metric_steps


if __name__ == '__main__':
    # presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    presets_cat = 'temporal'

    # eval_in_dir = os.path.join(DATA_DIR, 'eval_in')
    # prepare_ensemble_render_names(eval_in_dir, presets_cat)
    # exit()

    next_effect_architecture = next_effect_rnn.__name__
    # next_effect_architecture = next_effect_seq_only_rnn.__name__
    # next_effect_architecture = all_effects_cnn.__name__
    # next_effect_architecture = 'oracle_fixed_seq'
    # next_effect_architecture = 'oracle_random'
    # next_effect_architecture = 'fixed_seq'
    # next_effect_architecture = 'random'

    seq_effects = True
    # seq_effects = False

    mel_metrics = [mse, mae, mfcc_dist, lsd, pcc]
    audio_metrics = [mssmae]

    save_renders = True
    # save_renders = False

    # eval_mode = True
    eval_mode = False

    # test_rc_name = 'rendering/seq_5_v3_single.yaml'
    test_rc_name = 'rendering/seq_5_v3_train.yaml'
    rand_gen_n = 30

    # default_preset = 'sine'
    # default_preset = 'triangle'
    # default_preset = 'saw'
    # default_preset = 'square'

    # default_preset = 'sy_mtron_saw_[sd]'
    # default_preset = 'ld_power_5ths_[fp]'
    # default_preset = 'sy_shot_dirt_stab_[im]'
    # default_preset = 'sy_vintage_bells_[fp]'

    default_preset = 'sq_busy_lines_[lcv]'
    # default_preset = 'ld_iheardulike5ths_[sd]'
    # default_preset = 'ld_postmodern_talking_[fp]'
    # default_preset = 'sy_runtheharm_[gs]'

    model_dir = MODELS_DIR
    renders_save_dir = OUT_DIR

    target_effect_names_all = []
    effect_name_seq_all = []
    eval_metrics_all = defaultdict(list)
    n_completed = -1
    if eval_mode:
        eval_in_dir = os.path.join(DATA_DIR, 'eval_in')
        eval_in_path = os.path.join(eval_in_dir,
                                    f'seq_5_v3__mfcc_30__{presets_cat}'
                                    f'__ensemble__eval_in_data.npz')
        log.info(f'eval_in_path = {eval_in_path}')

        eval_in_data = np.load(eval_in_path)
        render_names = eval_in_data['render_names']
        presets = eval_in_data['presets']
        assert len(render_names) == len(presets)

        eval_out_dir = os.path.join(DATA_DIR, 'eval_out/ensemble')
        eval_save_name = f'seq_5_v3__mfcc_30__{presets_cat}__ensemble' \
                         f'__{next_effect_architecture}__is_seq_{seq_effects}' \
                         f'__eval_out_data.npz'
        eval_save_path = os.path.join(eval_out_dir, eval_save_name)
        log.info(f'eval_save_path = {eval_save_path}')

        if os.path.exists(eval_save_path):
            eval_data = np.load(eval_save_path, allow_pickle=True)
            target_effect_names_all = eval_data[
                'target_effect_names_all'].tolist()
            effect_name_seq_all = eval_data['effect_name_seq_all'].tolist()
            eval_metrics_all = eval_data['eval_metrics_all'].item()

            n_completed = len(target_effect_names_all)
            assert len(effect_name_seq_all) == n_completed
            assert all(len(v) == n_completed for v in eval_metrics_all.values())

        r_p = list(zip(render_names, presets))
    else:
        test_rc_path = os.path.join(CONFIGS_DIR, test_rc_name)
        with open(test_rc_path, 'r') as config_f:
            render_config = yaml.full_load(config_f)
        test_rc = RenderConfig(**render_config)
        test_rc_effects = test_rc.effects

        r_p = []
        for _ in range(rand_gen_n):
            rand_rc_effects = get_random_rc_effects(test_rc_effects,
                                                    min_n_effects=3,
                                                    max_n_effects=4,
                                                    use_all_effects=False)
            r_p.append((rand_rc_effects, default_preset))

    # results = crunch_eval_data(eval_save_path)
    # metric_name = 'mssmae'
    # log.info(f'{presets_cat}')
    # log.info(f'{next_effect_architecture}, {seq_effects}')
    # log.info(f'n\t{metric_name}')
    # for idx in range(6):
    #     curr_results = results[f'until_worse_{idx}']
    #     eval_v = curr_results['mean_d'][metric_name]
    #     if metric_name == 'mae' or metric_name == 'mse' or metric_name == 'pcc' or metric_name == 'mssmae':
    #         eval_v *= 100
    #     log.info(f'({idx}, {eval_v:>6.2f})')
    #
    # exit()

    next_effect_model_name = f'seq_5_v3__mfcc_30__{presets_cat}' \
                             f'__rnn__{next_effect_architecture}__best.h5'
    log.info(f'next_effect_model_name = {next_effect_model_name}')

    render_config_path = os.path.join(CONFIGS_DIR,
                                      'rendering/seq_5_v3_train.yaml')
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)
    rc = RenderConfig(**render_config)
    rc.use_hashes = False
    rc.preset = None

    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)

    next_effect_model_path = os.path.join(model_dir, next_effect_model_name)
    log.info(f'model_path = {next_effect_model_path}')

    if next_effect_architecture == next_effect_rnn.__name__:
        next_effect_model = load_model(next_effect_model_path)
        wrapper = NextEffectRNNWrapper(next_effect_model)
    elif next_effect_architecture == next_effect_seq_only_rnn.__name__:
        next_effect_model = load_model(next_effect_model_path)
        wrapper = NextEffectSeqOnlyRNNWrapper(next_effect_model)
    elif next_effect_architecture == all_effects_cnn.__name__:
        next_effect_model = load_model(next_effect_model_path)
        wrapper = AllEffectsCNNWrapper(next_effect_model)
    elif next_effect_architecture == 'oracle_fixed_seq':
        wrapper = OracleWrapper(is_fixed_seq=True)
    elif next_effect_architecture == 'oracle_random':
        wrapper = OracleWrapper(is_fixed_seq=False)
    elif next_effect_architecture == 'fixed_seq':
        wrapper = RandomWrapper(is_fixed_seq=True)
    else:
        wrapper = RandomWrapper(is_fixed_seq=False)

    effect_models = load_effect_cnns(model_dir,
                                     f'seq_5_v3__mfcc_30__{presets_cat}')
    log.info(f'n_completed = {n_completed}')

    for idx, (render_name, preset) in enumerate(r_p):
        if eval_mode and idx < n_completed:
            continue
        log.info('============================================================')
        log.info(f'next_effect_architecture = {next_effect_architecture}')
        log.info(f'seq_effects = {seq_effects}')
        log.info(f'idx = {idx}')
        log.info(f'preset = {preset}')
        log.info('')

        init_rc_effects = []
        if eval_mode:
            target_rc_effects = render_name_to_rc_effects(render_name)
        else:
            target_rc_effects = render_name

        preset_path = os.path.join(PRESETS_DIR, f'{preset}.fxp')
        renders_prefix = f'{next_effect_architecture}__{idx:03d}'

        target_effect_names, \
        effect_name_seq, \
        metric_steps = ensemble(init_rc_effects,
                                target_rc_effects,
                                preset_path,
                                rc,
                                pc,
                                renders_save_dir,
                                renders_prefix,
                                wrapper,
                                effect_models,
                                mel_metrics,
                                audio_metrics,
                                save_renders=save_renders,
                                seq_effects=seq_effects)

        if eval_mode:
            target_effect_names_all.append(target_effect_names)
            effect_name_seq_all.append(effect_name_seq)
            for k, v in metric_steps.items():
                eval_metrics_all[k].append(v)

            np.savez(eval_save_path,
                     target_effect_names_all=target_effect_names_all,
                     effect_name_seq_all=effect_name_seq_all,
                     eval_metrics_all=eval_metrics_all)
