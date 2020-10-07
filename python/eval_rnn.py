import logging
import os
from collections import defaultdict

import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf

from config import OUT_DIR, DATASETS_DIR, MODELS_DIR
from models import baseline_cnn
from training_rnn import EFFECT_TO_IDX_MAPPING, get_x_ids, RNNDataGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

tf.config.experimental.set_visible_devices([], 'GPU')

# EFFECT_TO_Y_PARAMS = {
#     'compressor': {270, 271, 272},
#     'distortion': {97, 99},
#     'eq': {88, 90, 92, 94},
#     'flanger': {105, 106, 107},
#     'phaser': {111, 112, 113, 114},
# }
EFFECT_TO_Y_PARAMS = {
    'compressor': {270, 271, 272},
    'distortion': {97, 99},
    'eq': {89, 91, 93},
    'phaser': {112, 113, 114},
    'reverb-hall': {81, 84, 86},
}


if __name__ == '__main__':
    in_x = 128
    in_y = 88
    n_channels = 2
    n_effects = len(EFFECT_TO_IDX_MAPPING)

    cnn_architecture = baseline_cnn
    # cnn_architecture = baseline_cnn_2x

    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    use_multiprocessing = False
    # use_multiprocessing = True
    workers = 8
    model_dir = MODELS_DIR

    model_name = f'seq_5_v3__{presets_cat}__rnn__{cnn_architecture.__name__}' \
                 f'__best.h5'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__rnn')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__{presets_cat}__rnn')
    model_path = os.path.join(model_dir, model_name)
    log.info(f'data_dir = {data_dir}')
    log.info(f'model_path = {model_path}')

    _, _, test_x_ids = get_x_ids(data_dir)
    # test_x_ids = test_x_ids[:100]
    log.info(f'test_x_ids length = {len(test_x_ids)}')

    test_gen = RNNDataGenerator(test_x_ids,
                                in_x,
                                in_y,
                                n_effects,
                                effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                                batch_size=1,
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

    results = defaultdict(list)
    for pred_row, ((_, effect_seq), y) in tqdm(zip(pred, test_gen)):
        n_effects = effect_seq.shape[1]
        pred_effect_idx = np.argmax(pred_row)
        actual_effect_idx = y[0]

        if pred_effect_idx == actual_effect_idx:
            results[n_effects].append(1)
        else:
            results[n_effects].append(0)

    log.info(f'model_name = {model_name}')
    log.info(f'eval_results = {eval_results}')
    log.info(f'pred.shape = {pred.shape}')

    all_results = []
    for n_effects, correct in sorted(results.items()):
        all_results.extend(correct)
        n = len(correct)
        correct_percentage = np.mean(correct)
        log.info(f'n_effects: {n_effects}, n = {n}, % = {correct_percentage}')

    log.info(f'all_results length = {len(all_results)}')
    log.info(f'all_results % = {np.mean(all_results)}')
    assert len(all_results) == len(pred)
