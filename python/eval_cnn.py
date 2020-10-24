import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import DATASETS_DIR, MODELS_DIR
from models_effect import baseline_cnn_2x
from training import get_x_y_metadata, DataGenerator, get_x_ids

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

tf.config.experimental.set_visible_devices([], 'GPU')


if __name__ == '__main__':
    effect = 'compressor'
    params = {270, 271, 272}
    # effect = 'distortion'
    # params = {97, 99}
    # effect = 'eq'
    # params = {89, 91, 93}
    # effect = 'phaser'
    # params = {112, 113, 114}
    # effect = 'reverb-hall'
    # params = {81, 84, 86}

    # architecture = baseline_cnn
    architecture = baseline_cnn_2x
    # architecture = baseline_cnn_shallow
    # architecture = exposure_cnn
    # architecture = baseline_lstm

    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    batch_size = 128
    channel_mode = 1
    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    model_dir = MODELS_DIR

    model_name = f'seq_5_v3__{presets_cat}__{effect}__{architecture.__name__}' \
                 f'__cm_{channel_mode}__best.h5'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__{effect}')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__{presets_cat}__{effect}')
    model_path = os.path.join(model_dir, model_name)
    log.info(f'data_dir = {data_dir}')
    log.info(f'model_path = {model_path}')

    x_y_metadata = get_x_y_metadata(data_dir, params)
    _, _, test_x_ids = get_x_ids(data_dir)
    # test_x_ids = test_x_ids[:256]
    log.info(f'test_x_ids length = {len(test_x_ids)}')

    test_gen = DataGenerator(test_x_ids,
                             x_y_metadata,
                             batch_size=batch_size,
                             channel_mode=channel_mode,
                             shuffle=False)

    model = load_model(model_path)
    model.summary()

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
