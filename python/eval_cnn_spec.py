import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

from audio_processing_util import ProcessConfig
from audio_rendering import RenderConfig
from config import PRESETS_DIR, MODELS_DIR, CONFIGS_DIR, OUT_DIR, DATA_DIR
from eval_util import effect_cnn_audio_step, create_effect_cnn_x, \
    render_name_to_rc_effects, get_patch_from_effect_cnn
from metrics import mae, mse, mfcc_dist, lsd, mssmae, pcc
from models_effect import baseline_cnn_2x
from training_util import EFFECT_TO_Y_PARAMS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

tf.config.experimental.set_visible_devices([], 'GPU')


if __name__ == '__main__':
    # presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    presets_cat = 'temporal'

    effect = 'compressor'
    # effect = 'distortion'
    # effect = 'eq'
    # effect = 'phaser'
    # effect = 'reverb-hall'
    params = EFFECT_TO_Y_PARAMS[effect]

    architecture = baseline_cnn_2x
    mel_metrics = [mse, mae, mfcc_dist, lsd, pcc]
    audio_metrics = [mssmae]

    # save_renders = True
    save_renders = False

    channel_mode = 1
    model_dir = MODELS_DIR

    model_name = f'seq_5_v3__mfcc_30__{presets_cat}__{effect}__' \
                 f'{architecture.__name__}__cm_{channel_mode}'
    log.info(f'model_name = {model_name}')

    eval_in_dir = os.path.join(DATA_DIR, 'eval_in')
    eval_in_path = os.path.join(eval_in_dir, f'{model_name}__eval_in_data.npz')
    log.info(f'eval_in_data_path = {eval_in_path}')

    eval_in_data = np.load(eval_in_path)
    mels = eval_in_data['mels']
    mfccs = eval_in_data['mfccs']
    render_names = eval_in_data['render_names']
    base_render_names = eval_in_data['base_render_names']
    presets = eval_in_data['presets']

    eval_out_dir = os.path.join(DATA_DIR, 'eval_out')
    eval_save_name = f'{model_name}__eval_out_data.npz'
    eval_save_path = os.path.join(eval_out_dir, eval_save_name)
    log.info(f'eval_save_path = {eval_save_path}')

    model_path = os.path.join(model_dir, f'{model_name}__best.h5')
    log.info(f'model_path = {model_path}')

    model = load_model(model_path)
    model.summary()

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

    x_base_mel_mses = []
    x_target_mel_mses = []
    dry_eval = defaultdict(list)
    wet_eval = defaultdict(list)
    n_completed = -1

    if os.path.exists(eval_save_path):
        eval_data = np.load(eval_save_path, allow_pickle=True)

        x_base_mel_mses = eval_data['x_base_mel_mses'].tolist()
        x_target_mel_mses = eval_data['x_target_mel_mses'].tolist()
        n_completed = len(x_base_mel_mses)
        assert len(x_target_mel_mses) == n_completed

        dry_eval = eval_data['dry_eval'].item()
        wet_eval = eval_data['wet_eval'].item()
        for _, data in dry_eval.items():
            assert len(data) == n_completed
        for _, data in wet_eval.items():
            assert len(data) == n_completed

    log.info(f'n_completed = {n_completed}')

    if n_completed > 0:
        log.info(f'model_name = {model_name}')
        log.info('')
        log.info(f'x_base_mel_mses   = {np.mean(x_base_mel_mses):.5f}')
        log.info(f'x_target_mel_mses = {np.mean(x_target_mel_mses):.5f}')
        log.info('')

        for metric_name, dry_v in dry_eval.items():
            wet_v = wet_eval[metric_name]
            dry_mean = np.mean(dry_v)
            wet_mean = np.mean(wet_v)
            delta_mean = wet_mean - dry_mean

            log.info(f'{metric_name:<9} dry   = {dry_mean:.5f}')
            log.info(f'{metric_name:<9} wet   = {wet_mean:.5f}')
            log.info(f'{metric_name:<9} delta = {delta_mean:.5f}')
    log.info('')

    for idx, (mel, mfcc, render_name, base_render_name, preset) in enumerate(
            zip(mels, mfccs, render_names, base_render_names, presets)):
        if idx < n_completed:
            continue
        log.info('')
        log.info(f'Current idx = {idx}')
        log.info('')

        preset_path = os.path.join(PRESETS_DIR, f'{preset}.fxp')
        log.info(f'Using preset {preset}')

        x_target_mel = mel[:, :, 0]
        x_base_mel = mel[:, :, 1]

        target_rc_effects = render_name_to_rc_effects(render_name)
        base_rc_effects = render_name_to_rc_effects(base_render_name)
        assert any(e['name'] == effect for e in target_rc_effects)
        assert all(e['name'] != effect for e in base_rc_effects)

        render_save_name = None

        if save_renders:
            render_save_name = f'{idx:03d}__target.wav'
        _, target_audio, target_af = effect_cnn_audio_step(
            preset_path,
            target_rc_effects,
            rc,
            pc,
            render_save_dir=OUT_DIR,
            render_save_name=render_save_name
        )

        if save_renders:
            render_save_name = f'{idx:03d}__dry.wav'
        engine, dry_audio, dry_af = effect_cnn_audio_step(
            preset_path,
            base_rc_effects,
            rc,
            pc,
            render_save_dir=OUT_DIR,
            render_save_name=render_save_name
        )

        cnn_x = create_effect_cnn_x(target_af, dry_af)
        cnn_pred = model.predict(cnn_x, batch_size=1)
        if not isinstance(cnn_pred, list):
            cnn_pred = [cnn_pred]
        cnn_pred[-1] = np.clip(cnn_pred[-1], a_min=0.0, a_max=1.0)

        patches = get_patch_from_effect_cnn(effect,
                                            cnn_pred,
                                            rc.gran,
                                            batch_size=1)
        patch = patches[0]
        log.info(f'pred patch = {patch}')
        step_rc_effects = [{'name': effect}]

        if save_renders:
            render_save_name = f'{idx:03d}__wet.wav'
        _, wet_audio, wet_af = effect_cnn_audio_step(
            preset_path,
            step_rc_effects,
            rc,
            pc,
            engine=engine,
            patch=patch,
            render_save_dir=OUT_DIR,
            render_save_name=render_save_name
        )

        x_target_mel_mse = mean_squared_error(x_target_mel, target_af.mel)
        x_base_mel_mse = mean_squared_error(x_base_mel, dry_af.mel)
        x_target_mel_mses.append(x_target_mel_mse)
        x_base_mel_mses.append(x_base_mel_mse)

        for metric in mel_metrics:
            dry_v = metric(dry_af.mel, target_af.mel)
            dry_eval[metric.__name__].append(dry_v)
            wet_v = metric(wet_af.mel, target_af.mel)
            wet_eval[metric.__name__].append(wet_v)
        for metric in audio_metrics:
            dry_result = metric(dry_audio, target_audio)
            for metric_name, dry_v in dry_result:
                dry_eval[metric_name].append(dry_v)
            wet_result = metric(wet_audio, target_audio)
            for metric_name, wet_v in wet_result:
                wet_eval[metric_name].append(wet_v)

        np.savez(eval_save_path,
                 x_base_mel_mses=x_base_mel_mses,
                 x_target_mel_mses=x_target_mel_mses,
                 dry_eval=dry_eval,
                 wet_eval=wet_eval)
