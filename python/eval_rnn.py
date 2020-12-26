import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import DATASETS_DIR, MODELS_DIR
from models_next_effect import next_effect_rnn, next_effect_seq_only_rnn, \
    all_effects_cnn
from training_rnn import EFFECT_TO_IDX_MAPPING, get_x_ids, RNNDataGenerator
from training_util import EffectSeqOnlyRNNDataGenerator, \
    AllEffectsCNNDataGenerator, IDX_TO_EFFECT_MAPPING

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
    # presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    presets_cat = 'temporal'

    in_x = 128
    in_y = 88
    n_mfcc = 30
    n_channels = 2
    n_effects = len(EFFECT_TO_IDX_MAPPING)

    architecture = next_effect_rnn
    # architecture = next_effect_seq_only_rnn
    # architecture = all_effects_cnn

    if architecture == next_effect_rnn:
        loss = 'sparse_categorical_crossentropy'
        metric = 'acc'
        data_gen = RNNDataGenerator
    elif architecture == next_effect_seq_only_rnn:
        loss = 'sparse_categorical_crossentropy'
        metric = 'acc'
        data_gen = EffectSeqOnlyRNNDataGenerator
    else:
        loss = 'binary_crossentropy'
        metric = 'binary_accuracy'
        data_gen = AllEffectsCNNDataGenerator

    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    model_dir = MODELS_DIR
    # model_dir = OUT_DIR

    # model_name = f'seq_5_v3_local__mfcc_30__{presets_cat}' \
    #              f'__rnn__{architecture.__name__}__best.h5'
    model_name = f'seq_5_v3__mfcc_30__{presets_cat}' \
                 f'__rnn__{architecture.__name__}__best.h5'
    log.info(f'model_name = {model_name}')

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'seq_5_v3_local__proc__{presets_cat}__rnn')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__proc__{presets_cat}__rnn')
    log.info(f'data_dir = {data_dir}')

    model_path = os.path.join(model_dir, model_name)
    log.info(f'model_path = {model_path}')

    _, _, test_x_ids = get_x_ids(data_dir)
    log.info(f'test_x_ids length = {len(test_x_ids)}')

    test_gen = data_gen(test_x_ids,
                        n_effects,
                        effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                        batch_size=1,
                        shuffle=False)

    model = architecture(in_x=in_x,
                         in_y=in_y,
                         n_mfcc=n_mfcc,
                         n_channels=n_channels,
                         n_effects=n_effects)
    model.summary()
    model.load_weights(model_path)
    log.info(f'loss = {loss}')
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metric)

    eval_results = model.evaluate(test_gen,
                                  use_multiprocessing=use_multiprocessing,
                                  workers=workers,
                                  return_dict=True,
                                  verbose=1)
    pred = model.predict(test_gen,
                         use_multiprocessing=use_multiprocessing,
                         workers=workers,
                         verbose=1)

    effect_results = defaultdict(list)
    seq_len_results = defaultdict(list)
    for pred_row, (x, y) in tqdm(zip(pred, test_gen)):
        if architecture == all_effects_cnn:
            y_list = [int(v) for v in y[0]]
            n_effects = sum(y_list)
            for y_idx, (pred_v, y_v) in enumerate(zip(pred_row, y_list)):
                actual_effect_name = IDX_TO_EFFECT_MAPPING[y_idx]
                pred_v = int(pred_v + 0.5)
                if pred_v == y_v:
                    effect_results[actual_effect_name].append(1)
                    seq_len_results[n_effects].append(1)
                else:
                    effect_results[actual_effect_name].append(0)
                    seq_len_results[n_effects].append(0)
        else:
            effect_seq = x[-1]
            n_effects = effect_seq.shape[1]
            pred_effect_idx = np.argmax(pred_row)
            actual_effect_idx = y[0]
            actual_effect_name = IDX_TO_EFFECT_MAPPING[actual_effect_idx]

            if pred_effect_idx == actual_effect_idx:
                effect_results[actual_effect_name].append(1)
                seq_len_results[n_effects].append(1)
            else:
                effect_results[actual_effect_name].append(0)
                seq_len_results[n_effects].append(0)

    log.info(f'model_name = {model_name}')
    log.info(f'eval_results = {eval_results}')
    log.info(f'pred.shape = {pred.shape}')
    log.info('')

    effect_all_results = []
    log.info('effect_results:')
    for effect_name, correct in sorted(effect_results.items()):
        effect_all_results.extend(correct)
        n = len(correct)
        correct_percent = np.mean(correct)
        log.info(f'effect_name: {effect_name:<11}, '
                 f'n = {n}, % = {correct_percent:.5f}')
    log.info('')
    log.info(f'effect_all_results length = {len(effect_all_results)}')
    log.info(f'effect_all_results % = {np.mean(effect_all_results):.5f}')
    log.info('')

    latex_table_row = []
    seq_len_all_results = []
    log.info('seq_len_results:')
    for n_effects, correct in sorted(seq_len_results.items()):
        seq_len_all_results.extend(correct)
        n = len(correct)
        correct_percent = np.mean(correct)
        log.info(f'n_effects: {n_effects}, n = {n}, % = {correct_percent:.5f}')
        latex_table_row.append(f'{correct_percent:>5.3f}')
    log.info('')
    log.info(f'seq_len_all_results length = {len(seq_len_all_results)}')
    log.info(f'seq_len_all_results % = {np.mean(seq_len_all_results):.5f}')
    latex_table_row.append(f'{np.mean(seq_len_all_results):>5.3f}')

    assert len(effect_all_results) == len(seq_len_all_results)
    if architecture != all_effects_cnn:
        assert len(effect_all_results) == len(pred)
    log.info('')
    log.info(f'latex table row = {" & ".join(latex_table_row)}')
