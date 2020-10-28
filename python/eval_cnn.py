import logging
import os

import numpy as np
import tensorflow as tf

from config import DATASETS_DIR, MODELS_DIR
from models_effect import baseline_cnn_2x, build_effect_model
from training import get_x_y_metadata, DataGenerator, get_x_ids
from training_util import EFFECT_TO_Y_PARAMS

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')
    # tf.config.experimental.set_visible_devices([], 'GPU')


if __name__ == '__main__':
    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    effect = 'compressor'
    # effect = 'distortion'
    # effect = 'eq'
    # effect = 'phaser'
    # effect = 'reverb-hall'
    params = EFFECT_TO_Y_PARAMS[effect]

    architecture = baseline_cnn_2x

    batch_size = 128
    channel_mode = 1
    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    model_dir = MODELS_DIR

    model_name = f'seq_5_v3__mfcc_30__{presets_cat}__{effect}__' \
                 f'{architecture.__name__}__cm_{channel_mode}__best.h5'
    log.info(f'model_name = {model_name}')

    datasets_dir = DATASETS_DIR
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__proc__{presets_cat}'
                                          f'__{effect}')
    log.info(f'data_dir = {data_dir}')

    model_path = os.path.join(model_dir, model_name)
    log.info(f'model_path = {model_path}')

    x_y_metadata = get_x_y_metadata(data_dir, params)
    _, _, test_x_ids = get_x_ids(data_dir)
    log.info(f'batch_size = {batch_size}')

    test_gen = DataGenerator(test_x_ids,
                             x_y_metadata,
                             batch_size=batch_size,
                             channel_mode=channel_mode,
                             shuffle=False)

    model = build_effect_model(x_y_metadata.in_x,
                               x_y_metadata.in_y,
                               n_mfcc=x_y_metadata.n_mfcc,
                               architecture=architecture,
                               n_bin=x_y_metadata.n_bin,
                               n_cate=x_y_metadata.n_cate,
                               cate_names=x_y_metadata.cate_names,
                               n_cont=x_y_metadata.n_cont,
                               use_fast_data_gen=False)
    model.summary()
    model.load_weights(model_path)

    model.compile(optimizer='adam',
                  loss=x_y_metadata.y_losses,
                  metrics=x_y_metadata.metrics)

    eval_results = model.evaluate(test_gen,
                                  use_multiprocessing=use_multiprocessing,
                                  workers=workers,
                                  return_dict=True,
                                  verbose=1)
    pred = model.predict(test_gen,
                         use_multiprocessing=use_multiprocessing,
                         workers=workers,
                         verbose=1)
    if not isinstance(pred, list):
        pred = [pred]

    log.info(f'model_name = {model_name}')
    log.info(f'test_x_ids length = {len(test_x_ids)}')
    log.info(f'eval_results = {eval_results}')

    if x_y_metadata.n_cont:
        y_cont = np.concatenate([y[-1] for _, y in test_gen], axis=0)
        log.info(f'y_cont shape = {y_cont.shape}')
        pred_cont = pred[-1]
        mse = np.square(y_cont - pred_cont)
        mse = np.mean(mse, axis=0)
        log.info(f'y_cont granular MSE = {mse}')
        log.info(f'y_cont overall MSE = {np.mean(mse)}')
        mae = np.abs(y_cont - pred_cont)
        mae = np.mean(mae, axis=0)
        log.info(f'y_cont granular MAE = {mae}')
        log.info(f'y_cont overall MAE = {np.mean(mae)}')
