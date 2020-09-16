import logging
import os
from collections import namedtuple
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import DATA_DIR, OUT_DIR
from models import build_effect_model, exposure_cnn, baseline_cnn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[GPU], enable=True)

YData = namedtuple(
    'YData',
    'n_bin n_cate cate_names n_cont y_s y_losses metrics'
)


def train_model(
        model: Model,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        model_name: str,
        batch_size: int = 512,
        epochs: int = 128,
        val_split: float = 0.20,
        patience: int = 16,
        output_dir_path: str = OUT_DIR) -> None:
    save_path = os.path.join(
        output_dir_path,
        # model_name + '_e{epoch:03d}_vl{val_loss:.4f}.h5'
        f'{model_name}_best.h5'
    )
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=1)
    cp = ModelCheckpoint(save_path,
                         monitor='val_loss',
                         save_best_only=True,
                         verbose=1)
    model.fit(x,
              y,
              shuffle=True,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=val_split,
              callbacks=[es, cp],
              verbose=1)


def prepare_y_data(y_data_path: str) -> YData:
    y_npz_data = np.load(y_data_path)

    n_bin = 0
    n_cate = []
    cate_names = []
    n_cont = 0
    y_s = []
    y_losses = {}
    metrics = {}

    if 'binary_params' in y_npz_data:
        bin_params = y_npz_data['binary_params'].tolist()
        n_bin = len(bin_params)
        y_bin = y_npz_data['binary']
        log.info(f'y_bin shape = {y_bin.shape}')
        assert y_bin.shape[-1] == n_bin
        y_s.append(y_bin)
        y_losses['bin_output'] = 'bce'
        metrics['bin_output'] = 'acc'

    if 'categorical_params' in y_npz_data:
        n_cate = y_npz_data['n_categories'].tolist()
        descs = y_npz_data['param_to_desc'].tolist()

        for n_classes, desc in zip(n_cate, descs):
            y_cate = y_npz_data[desc]
            cate_name = desc.strip().lower().replace(' ', '_')
            log.info(f'{cate_name} n_classes = {n_classes}')
            log.info(f'{cate_name} max = {np.max(y_cate)}')
            log.info(f'{cate_name} min = {np.min(y_cate)}')
            assert 0 <= np.max(y_cate) < n_classes
            assert 0 <= np.min(y_cate) < n_classes
            cate_names.append(cate_name)
            y_s.append(y_cate)
            y_losses[cate_name] = 'sparse_categorical_crossentropy'
            metrics[cate_name] = 'acc'

    if 'continuous_params' in y_npz_data:
        cont_params = y_npz_data['continuous_params'].tolist()
        n_cont = len(cont_params)
        y_cont = y_npz_data['continuous']
        assert y_cont.shape[-1] == n_cont
        log.info(f'y_cont shape = {y_cont.shape}')
        y_s.append(y_cont)
        y_losses['cont_output'] = 'mse'
        metrics['cont_output'] = 'mae'

    log.info(f'y_losses = {y_losses}')
    log.info(f'metrics = {metrics}')

    y_data = YData(n_bin=n_bin,
                   n_cate=n_cate,
                   cate_names=cate_names,
                   n_cont=n_cont,
                   y_s=y_s,
                   y_losses=y_losses,
                   metrics=metrics)

    return y_data


if __name__ == '__main__':
    # n = 14014
    n = 25000
    # gran = 1000
    gran = 100
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
    effect = 'reverb_hall'
    params = {82, 83, 84, 85, 86, 87}

    architecture = baseline_cnn
    # architecture = exposure_cnn
    batch_size = 512
    epochs = 128
    val_split = 0.20
    patience = 16
    model_name = f'{effect}_{architecture.__name__}'

    params = sorted([str(_) for _ in params])
    params = '_'.join(params)

    x_data_path = f'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}.npz'
    x_data_path = os.path.join(DATA_DIR, x_data_path)
    x_npz_data = np.load(x_data_path)
    x = x_npz_data['mels']
    in_x = x.shape[1]
    in_y = x.shape[2]
    log.info(f'mels shape = {x.shape}')
    log.info(f'in_x = {in_x}, in_y = {in_y}')

    y_data_path = f'{os.path.splitext(x_data_path)[0]}__y_{params}.npz'
    y_data = prepare_y_data(y_data_path)

    model = build_effect_model(in_x,
                               in_y,
                               architecture=architecture,
                               n_bin=y_data.n_bin,
                               n_cate=y_data.n_cate,
                               cate_names=y_data.cate_names,
                               n_cont=y_data.n_cont)

    model.compile(optimizer='adam',
                  loss=y_data.y_losses,
                  metrics=y_data.metrics)
    model.summary()

    train_model(model,
                x,
                y_data.y_s,
                model_name,
                batch_size=batch_size,
                epochs=epochs,
                val_split=val_split,
                patience=patience)
