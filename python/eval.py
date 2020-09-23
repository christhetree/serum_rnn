import logging
import os

import numpy as np
import soundfile as sf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tqdm import tqdm

from audio_processing import ProcessConfig, parse_save_name, get_mel_spec
from audio_rendering import RenderConfig, PatchGenerator, generate_render_hash, \
    render_patch
from config import MODELS_DIR, DATA_DIR, PRESETS_DIR
from effects import param_to_effect, PARAM_TO_DESC, get_effect
from models import baseline_cnn
from serum_util import setup_serum, set_preset
from training import prepare_y_model_data

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def eval_model(model_path: str,
               x_data_path: str,
               y_data_path: str,
               batch_size: int = 512,
               reg_eval: bool = True,
               granular_eval: bool = True,
               mel_eval: bool = True,
               save_pred_renders: bool = True) -> None:
    x_npz_data = np.load(x_data_path, allow_pickle=True)
    x = x_npz_data['mels']
    in_x = x.shape[1]
    in_y = x.shape[2]
    log.info(f'mels shape = {x.shape}')
    log.info(f'in_x = {in_x}, in_y = {in_y}')

    y_model_data = prepare_y_model_data(y_data_path)
    y = y_model_data.y_s

    model: Model = load_model(model_path)
    model.compile(optimizer='adam',
                  loss=y_model_data.y_losses,
                  metrics=y_model_data.metrics)
    model.summary()

    if reg_eval:
        log.info(f'Starting regular evaluation:')
        reg_eval_result = model.evaluate(x,
                                         y,
                                         batch_size=batch_size,
                                         return_dict=True,
                                         verbose=1)
        log.info(reg_eval_result)

    if not granular_eval and not mel_eval:
        return

    pred = model.predict(x, batch_size=batch_size, verbose=1)
    if not isinstance(pred, list):
        pred = [pred]

    if granular_eval:
        log.info(f'Starting granular evaluation:')

        # # TODO
        # if y_data.n_bin:
        #     y_bin = y_data.y_s[0]
        #     pred_bin = pred[0]
        #     mae = np.abs(y_bin - pred_bin)
        #     mae = np.mean(mae, axis=0)

        if y_model_data.n_cont:
            y_cont = y_model_data.y_s[-1]
            pred_cont = pred[-1]
            mae = np.abs(y_cont - pred_cont)
            mae = np.mean(mae, axis=0)
            log.info(f'y_cont granular MAE = {mae}')  # TODO

    if mel_eval:
        log.info(f'Starting spectrogram evaluation:')
        log.info(f'save_pred_renders = {save_pred_renders}')
        y_npz_data = np.load(y_data_path, allow_pickle=True)

        pc = ProcessConfig(**x_npz_data['process_config'].item())
        log.info(f'process config = {pc}')
        dirs = os.path.normpath(pc.root_dir).split(os.path.sep)
        effect_dir_name = dirs[-1]
        int_dir_name = dirs[-2]

        effect_dir_info = parse_save_name(effect_dir_name, is_dir=True)
        assert effect_dir_info['gran'] == y_npz_data['gran'].item()

        int_dir_info = parse_save_name(int_dir_name, is_dir=True)
        preset_path = os.path.join(PRESETS_DIR, f'{int_dir_info["name"]}.fxp')
        rc = RenderConfig(sr=int_dir_info['sr'],
                          note_length=int_dir_info['nl'],
                          render_length=int_dir_info['rl'],
                          vel=int_dir_info['vel'],
                          midi=int_dir_info['midi'],
                          gran=effect_dir_info['gran'],
                          preset=preset_path)
        log.info(f'render config = {rc.__dict__}')

        engine = setup_serum(rc.preset, sr=rc.sr, render_once=True)

        pred_y_s = pred
        pred_bin = None
        pred_cate_s = None
        pred_cont = None

        if y_model_data.n_bin:
            pred_bin = pred_y_s[0]
            pred_y_s = pred_y_s[1:]
            pred_bin = np.around(pred_bin).astype(np.int32)

        if y_model_data.n_cont:
            pred_cont = pred_y_s[-1]
            pred_y_s = pred_y_s[:-1]
            pred_cont = np.around(pred_cont * rc.gran).astype(np.int32)

        if y_model_data.n_cate:
            pred_cate_s = [np.argmax(_, axis=-1) for _ in pred_y_s]

        rc_effects = {}
        bin_params = y_npz_data['binary_params'] \
            if 'binary_params' in y_npz_data else []
        cate_params = y_npz_data['categorical_params'] \
            if 'categorical_params' in y_npz_data else []
        cont_params = y_npz_data['continuous_params'] \
            if 'continuous_params' in y_npz_data else []

        mel_mses = []
        mel_maes = []
        n_perfect_pred = 0
        render_names = x_npz_data['render_names']

        pred_save_dir = os.path.join(pc.root_dir, 'eval')
        if save_pred_renders and not os.path.exists(pred_save_dir):
            log.info('Creating eval folder.')
            os.makedirs(pred_save_dir)

        for idx in tqdm(range(len(x))):
            render_name = render_names[idx]
            mel = x[idx].squeeze()

            if pred_bin is not None:
                pred_bin_row = pred_bin[idx]
                for bin_param, pred_bin_v in zip(bin_params, pred_bin_row):
                    effect_name = param_to_effect[bin_param].name
                    if effect_name not in rc_effects:
                        rc_effects[effect_name] = {'name': effect_name}
                    rc_effect = rc_effects[effect_name]
                    desc = PARAM_TO_DESC[bin_param]
                    rc_effect[desc] = [pred_bin_v]

            if pred_cate_s is not None:
                for cate_param, pred_cate in zip(cate_params, pred_cate_s):
                    pred_cate_v = pred_cate[idx]
                    effect_name = param_to_effect[cate_param].name
                    if effect_name not in rc_effects:
                        rc_effects[effect_name] = {'name': effect_name}
                    rc_effect = rc_effects[effect_name]
                    desc = PARAM_TO_DESC[cate_param]
                    rc_effect[desc] = [pred_cate_v]

            if pred_cont is not None:
                pred_cont_row = pred_cont[idx]
                for cont_param, pred_cont_v in zip(cont_params, pred_cont_row):
                    effect_name = param_to_effect[cont_param].name
                    if effect_name not in rc_effects:
                        rc_effects[effect_name] = {'name': effect_name}
                    rc_effect = rc_effects[effect_name]
                    desc = PARAM_TO_DESC[cont_param]
                    rc_effect[desc] = [pred_cont_v]

            rc.effects = list(rc_effects.values())
            pg = PatchGenerator(rc)
            assert pg.n_combos == 1

            for effect_render_data in rc.effects:
                effect = get_effect(effect_render_data['name'])
                set_preset(engine, effect.default)

            default_diff, patch = list(pg.generate_all_combos())[0]
            pred_render_name = generate_render_hash(pg.effect_names,
                                                    default_diff,
                                                    pg.param_n_digits)

            if render_name == pred_render_name:
                n_perfect_pred += 1

            pred_audio = render_patch(engine,
                                      patch,
                                      rc,
                                      save_dir=None,
                                      render_name=None)

            pred_mel = get_mel_spec(pred_audio,
                                    sr=pc.sr,
                                    hop_length=pc.hop_length,
                                    n_mels=pc.n_mels,
                                    n_fft=pc.n_fft,
                                    max_n_of_frames=pc.max_n_of_frames,
                                    normalize_audio=pc.normalize_audio,
                                    normalize_mel=pc.normalize_mel)

            mse = mean_squared_error(mel, pred_mel)
            mel_mses.append(mse)
            mae = mean_absolute_error(mel, pred_mel)
            mel_maes.append(mae)

            if save_pred_renders:
                save_render_name = f'{render_name}___{pred_render_name}' \
                                   f'___mse_{mse:.4f}__mae_{mae:.4f}.wav'
                save_path = os.path.join(pred_save_dir, save_render_name)
                sf.write(save_path, pred_audio, rc.sr)


        mel_mses = np.array(mel_mses)
        mel_maes = np.array(mel_maes)
        log.info(f'n_perfect_pred = {n_perfect_pred}')
        log.info(f'perfect pred = {(n_perfect_pred / len(x) * 100):.2f}%')
        log.info(f'mean mse = {np.mean(mel_mses).item():.4f}')
        log.info(f'std mse = {np.std(mel_mses).item():.4f}')
        log.info(f'mean mae = {np.mean(mel_maes).item():.4f}')
        log.info(f'std mae = {np.std(mel_maes).item():.4f}')


if __name__ == '__main__':
    # n = 14014
    # n = 56
    n = 1000
    # n = 25000
    # gran = 1000
    gran = 100
    effect = 'chorus'
    params = {118, 119, 120, 121, 122, 123}
    # effect = 'distortion'
    # params = {97, 99}
    # effect = 'eq'
    # params = {88, 89, 90, 91, 92, 93, 94, 95}
    # effect = 'filter'
    # params = {142, 143, 144, 145, 146, 268}
    # effect = 'flanger'
    # params = {105, 106, 107, 108}
    # effect = 'phaser'
    # params = {111, 112, 113, 114, 115}
    # effect = 'reverb-hall'
    # params = {82, 83, 84, 85, 86, 87}

    architecture = baseline_cnn
    # architecture = exposure_cnn
    batch_size = 512
    model_name = f'{effect}_{architecture.__name__}_best.h5'
    model_path = os.path.join(MODELS_DIR, model_name)

    params = sorted([str(_) for _ in params])
    params = '_'.join(params)

    # x_data_path = f'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}.npz'
    x_data_path = f'audio_test_data/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}.npz'
    x_data_path = os.path.join(DATA_DIR, x_data_path)
    y_data_path = f'{os.path.splitext(x_data_path)[0]}__y_{params}.npz'

    eval_model(model_path,
               x_data_path,
               y_data_path,
               batch_size=batch_size)
