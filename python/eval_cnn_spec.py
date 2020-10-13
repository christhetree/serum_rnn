import logging
import os

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

from audio_processing_util import ProcessConfig, get_mel_spec
from audio_rendering import RenderConfig, render_patch
from config import PRESETS_DIR, MODELS_DIR, DATASETS_DIR, CONFIGS_DIR, OUT_DIR
from eval_ensemble import set_default_and_constant_params, \
    get_patch_from_effect_cnn, render_name_to_rc_effects
from metrics import calc_lsd, calc_mfcc_metric
from models import baseline_cnn_2x
from serum_util import setup_serum

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

tf.config.experimental.set_visible_devices([], 'GPU')


if __name__ == '__main__':
    # effect = 'compressor'
    # effect = 'distortion'
    # effect = 'eq'
    # effect = 'phaser'
    effect = 'reverb-hall'

    architecture = baseline_cnn_2x

    # presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    presets_cat = 'temporal'

    batch_size = 1
    channel_mode = 1
    use_multiprocessing = False
    # use_multiprocessing = True
    workers = 8
    model_dir = MODELS_DIR

    model_name = f'seq_5_v3__{presets_cat}__{effect}__{architecture.__name__}' \
                 f'__cm_{channel_mode}'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__{effect}')
    data_path = os.path.join(datasets_dir, f'{model_name}__eval_spec_data.npz')
    data = np.load(data_path)
    x_s = data['x_s']
    render_names = data['render_names']
    base_render_names = data['base_render_names']
    presets = data['presets']

    model_path = os.path.join(model_dir, f'{model_name}__best.h5')
    log.info(f'data_dir = {data_path}')
    log.info(f'model_path = {model_path}')

    model = load_model(model_path)
    model.summary()

    render_config_path = os.path.join(CONFIGS_DIR,
                                      'rendering/seq_5_v3_train.yaml')
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)
    rc = RenderConfig(**render_config)
    rc.use_hashes = False

    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)

    x_target_mel_mses = []
    x_base_mel_mses = []
    base_mses = []
    base_maes = []
    base_mfcc_dists = []
    base_lsds = []
    mses = []
    maes = []
    mfcc_dists = []
    lsds = []

    save_name = f'{model_name}__eval_data.npz'
    save_path = os.path.join(datasets_dir, save_name)
    if os.path.exists(save_path):
        eval_data = np.load(save_path)
        n_points = -1
        for _, data in eval_data.items():
            if n_points == -1:
                n_points = len(data)
            else:
                assert len(data) == n_points

        x_target_mel_mses = eval_data['x_target_mel_mses'].tolist()
        x_base_mel_mses = eval_data['x_base_mel_mses'].tolist()
        base_mses = eval_data['base_mses'].tolist()
        base_maes = eval_data['base_maes'].tolist()
        base_mfcc_dists = eval_data['base_mfcc_dists'].tolist()
        base_lsds = eval_data['base_lsds'].tolist()
        mses = eval_data['mses'].tolist()
        maes = eval_data['maes'].tolist()
        mfcc_dists = eval_data['mfcc_dists'].tolist()
        lsds = eval_data['lsds'].tolist()

    completed_n = len(maes)
    log.info(f'completed_n = {completed_n}')

    if completed_n > 0:
        log.info(f'x_target_mel_mses mean = {np.mean(x_target_mel_mses)}')
        log.info(f'x_base_mel_mses mean = {np.mean(x_base_mel_mses)}')
        log.info(f'base_mses mean = {np.mean(base_mses)}')
        log.info(f'mses mean = {np.mean(mses)}')
        log.info(f'base_mfcc_dists mean = {np.mean(base_mfcc_dists)}')
        log.info(f'mfcc_dists mean = {np.mean(mfcc_dists)}')
        log.info(f'base_lsds mean = {np.mean(base_lsds)}')
        log.info(f'lsds mean = {np.mean(lsds)}')

        log.info('')
        log.info(f'mse  delta = {np.mean(mses) - np.mean(base_mses):.3f}')
        log.info(f'mae  delta = {np.mean(maes) - np.mean(base_maes):.3f}')
        log.info(f'mfcc delta = {np.mean(mfcc_dists) - np.mean(base_mfcc_dists):.2f}')
        log.info(f'lsd  delta = {np.mean(lsds) - np.mean(base_lsds):.2f}')

    # exit()

    for idx, (x, render_name, base_render_name, preset) in enumerate(
            zip(x_s, render_names, base_render_names, presets)):
        if idx < completed_n:
            continue

        x_target_mel = x[:, :, 0]
        x_base_mel = x[:, :, 1]
        preset_path = os.path.join(PRESETS_DIR, f'subset/{preset}.fxp')
        assert os.path.exists(preset_path)
        log.info(f'Using preset {preset}')
        # assert preset in {'sine', 'triangle', 'saw', 'square'}
        # assert preset in {'ld_power_5ths_[fp]', 'sy_mtron_saw_[sd]',
        #                   'sy_shot_dirt_stab_[im]', 'sy_vintage_bells_[fp]'}
        assert preset in {'ld_iheardulike5ths_[sd]', 'ld_postmodern_talking_[fp]', 'sq_busy_lines_[lcv]', 'sy_runtheharm_[gs]'}

        target_rc_effects = render_name_to_rc_effects(render_name)
        base_rc_effects = render_name_to_rc_effects(base_render_name)
        assert any(e['name'] == effect for e in target_rc_effects)
        assert all(e['name'] != effect for e in base_rc_effects)

        target_engine = setup_serum(preset_path, sr=rc.sr, render_once=True, instance=1)
        set_default_and_constant_params(target_engine,
                                        target_rc_effects,
                                        rc.effects,
                                        gran=rc.gran)
        target_audio = render_patch(target_engine,
                                    {},
                                    rc,
                                    OUT_DIR,
                                    f'target_{idx}.wav')
        target_mel = get_mel_spec(target_audio,
                                  pc.sr,
                                  hop_length=pc.hop_length,
                                  n_mels=pc.n_mels,
                                  n_fft=pc.n_fft,
                                  max_n_of_frames=pc.max_n_of_frames,
                                  normalize_audio=pc.normalize_audio,
                                  normalize_mel=pc.normalize_mel)

        engine = setup_serum(preset_path, sr=rc.sr, render_once=True, instance=2)
        set_default_and_constant_params(engine,
                                        base_rc_effects,
                                        rc.effects,
                                        gran=rc.gran)
        base_audio = render_patch(engine,
                                  {},
                                  rc,
                                  OUT_DIR,
                                  f'base_{idx}.wav')
        base_mel = get_mel_spec(base_audio,
                                pc.sr,
                                hop_length=pc.hop_length,
                                n_mels=pc.n_mels,
                                n_fft=pc.n_fft,
                                max_n_of_frames=pc.max_n_of_frames,
                                normalize_audio=pc.normalize_audio,
                                normalize_mel=pc.normalize_mel)

        cnn_x = np.stack([target_mel, base_mel], axis=-1)
        cnn_x = np.expand_dims(cnn_x, axis=0)
        cnn_pred = model.predict(cnn_x, batch_size=1)
        if not isinstance(cnn_pred, list):
            cnn_pred = [cnn_pred]

        patches = get_patch_from_effect_cnn(effect,
                                            cnn_pred,
                                            rc.gran,
                                            batch_size=1)
        patch = patches[0]
        log.info(f'pred patch = {patch}')

        set_default_and_constant_params(engine,
                                        [{'name': effect}],
                                        rc.effects,
                                        gran=rc.gran)
        pred_audio = render_patch(engine,
                                  patch,
                                  rc,
                                  OUT_DIR,
                                  f'pred_{idx}.wav')
        pred_mel = get_mel_spec(pred_audio,
                                pc.sr,
                                hop_length=pc.hop_length,
                                n_mels=pc.n_mels,
                                n_fft=pc.n_fft,
                                max_n_of_frames=pc.max_n_of_frames,
                                normalize_audio=pc.normalize_audio,
                                normalize_mel=pc.normalize_mel)

        x_target_mel_mse = mean_squared_error(x_target_mel, target_mel)
        x_base_mel_mse = mean_squared_error(x_base_mel, base_mel)

        base_mse = mean_squared_error(target_mel, base_mel)
        base_mae = mean_absolute_error(target_mel, base_mel)
        base_mfcc_dist = calc_mfcc_metric(target_mel,
                                          base_mel,
                                          n_mfcc=20,
                                          sr=pc.sr,
                                          normalized_mel=pc.normalize_mel)
        base_lsd = calc_lsd(target_mel,
                            base_mel,
                            normalized_mel=pc.normalize_mel)

        mse = mean_squared_error(target_mel, pred_mel)
        mae = mean_absolute_error(target_mel, pred_mel)
        mfcc_dist = calc_mfcc_metric(target_mel,
                                     pred_mel,
                                     n_mfcc=20,
                                     sr=pc.sr,
                                     normalized_mel=pc.normalize_mel)
        lsd = calc_lsd(target_mel, pred_mel, normalized_mel=pc.normalize_mel)

        x_target_mel_mses.append(x_target_mel_mse)
        x_base_mel_mses.append(x_base_mel_mse)
        base_mses.append(base_mse)
        base_maes.append(base_mae)
        base_mfcc_dists.append(base_mfcc_dist)
        base_lsds.append(base_lsd)
        mses.append(mse)
        maes.append(mae)
        mfcc_dists.append(mfcc_dist)
        lsds.append(lsd)

        np.savez(save_path,
                 x_target_mel_mses=x_target_mel_mses,
                 x_base_mel_mses=x_base_mel_mses,
                 base_mses=base_mses,
                 base_maes=base_maes,
                 base_mfcc_dists=base_mfcc_dists,
                 base_lsds=base_lsds,
                 mses=mses,
                 maes=maes,
                 mfcc_dists=mfcc_dists,
                 lsds=lsds)

        log.info(f'x_target_mel_mse = {x_target_mel_mse:.6f}')
        log.info(f'x_base_mel_mse = {x_base_mel_mse:.6f}')
        log.info(f'base_mse = {base_mse:.6f}')
        log.info(f'base_mae = {base_mae:.6f}')
        log.info(f'base_mfcc_metric = {base_mfcc_dist:.6f}')
        log.info(f'base_lsd = {base_lsd:.6f}')
        log.info(f'mse = {mse:.6f}')
        log.info(f'mae = {mae:.6f}')
        log.info(f'mfcc_metric = {mfcc_dist:.6f}')
        log.info(f'lsd = {lsd:.6f}')
