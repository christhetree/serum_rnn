import logging
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from config import OUT_DIR, DATASETS_DIR
from models_next_effect import next_effect_rnn, next_effect_seq_only_rnn, \
    all_effects_cnn
from training_util import RNNDataGenerator, EFFECT_TO_IDX_MAPPING, \
    EffectSeqOnlyRNNDataGenerator, AllEffectsCNNDataGenerator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')


def train_model_gen(model: Model,
                    train_gen: RNNDataGenerator,
                    val_gen: RNNDataGenerator,
                    model_name: str,
                    epochs: int = 100,
                    patience: int = 8,
                    min_delta: float = 0.0001,
                    output_dir_path: str = OUT_DIR,
                    use_multiprocessing: bool = True,
                    workers: int = 8) -> None:
    save_path = os.path.join(
        output_dir_path,
        # model_name + '_e{epoch:03d}_vl{val_loss:.4f}.h5'
        f'{model_name}__best.h5'
    )
    es = EarlyStopping(monitor='val_loss',
                       min_delta=min_delta,
                       patience=patience,
                       verbose=1)
    cp = ModelCheckpoint(save_path,
                         monitor='val_loss',
                         save_best_only=True,
                         verbose=1)
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=epochs,
              callbacks=[es, cp],
              use_multiprocessing=use_multiprocessing,
              workers=workers,
              verbose=1)


def get_x_ids(data_dir: str,
              val_split: float = 0.10,
              test_split: float = 0.05,
              max_n: int = -1,
              use_cached: bool = True) -> (List[Tuple[List[str], List[str]]],
                                           List[Tuple[List[str], List[str]]],
                                           List[Tuple[List[str], List[str]]]):
    assert val_split + test_split < 1.0
    train_x_ids_path = os.path.join(data_dir, 'train_x_ids.npy')
    val_x_ids_path = os.path.join(data_dir, 'val_x_ids.npy')
    test_x_ids_path = os.path.join(data_dir, 'test_x_ids.npy')

    if use_cached \
        and all(os.path.exists(p)
                for p in [train_x_ids_path, val_x_ids_path, test_x_ids_path]):
        log.info('Using cached x_ids.')
        train_x_ids = np.load(train_x_ids_path, allow_pickle=True)
        val_x_ids = np.load(val_x_ids_path, allow_pickle=True)
        test_x_ids = np.load(test_x_ids_path, allow_pickle=True)
    else:
        x_ids = []
        for npz_name in tqdm(os.listdir(data_dir)):
            if not npz_name.endswith('.npz'):
                continue

            npz_data = np.load(os.path.join(data_dir, npz_name))
            mel_path_seq = npz_data['mel_path_seq'].tolist()
            effect_seq = npz_data['effect_seq'].tolist()
            x_ids.append((mel_path_seq, effect_seq))

        log.info(f'Found {len(x_ids)} data points.')

        np.random.shuffle(x_ids)
        if max_n > 0:
            x_ids = x_ids[:max_n]

        val_idx = int(len(x_ids) * (1.0 - val_split - test_split))
        test_idx = int(len(x_ids) * (1.0 - test_split))

        train_x_ids = x_ids[:val_idx]
        val_x_ids = x_ids[val_idx:test_idx]
        test_x_ids = x_ids[test_idx:]

        log.info('Caching x_ids.')
        np.save(train_x_ids_path, train_x_ids)
        np.save(val_x_ids_path, val_x_ids)
        np.save(test_x_ids_path, test_x_ids)

    return train_x_ids, val_x_ids, test_x_ids


if __name__ == '__main__':
    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    in_x = 128
    in_y = 88
    n_mfcc = 30
    n_channels = 2
    n_effects = len(EFFECT_TO_IDX_MAPPING)

    architecture = next_effect_rnn
    # architecture = next_effect_seq_only_rnn
    # architecture = all_effects_cnn

    if architecture == next_effect_rnn:
        batch_size = 32
        loss = 'sparse_categorical_crossentropy'
        metric = 'acc'
        data_gen = RNNDataGenerator
    elif architecture == next_effect_seq_only_rnn:
        batch_size = 32
        loss = 'sparse_categorical_crossentropy'
        metric = 'acc'
        data_gen = EffectSeqOnlyRNNDataGenerator
    else:
        batch_size = 32
        loss = 'binary_crossentropy'
        metric = 'binary_accuracy'
        data_gen = AllEffectsCNNDataGenerator

    epochs = 100
    val_split = 0.10
    test_split = 0.05
    patience = 8
    min_delta = 0.0001
    used_cached_x_ids = True
    max_n = -1
    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    load_prev_model = False

    # model_name = f'seq_5_v3_local__mfcc_30__{presets_cat}__rnn__{architecture.__name__}'
    model_name = f'seq_5_v3__mfcc_30__{presets_cat}' \
                 f'__rnn__{architecture.__name__}'
    log.info(f'model_name = {model_name}')

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'seq_5_v3_local__proc__{presets_cat}__rnn')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__proc__{presets_cat}__rnn')
    log.info(f'data_dir = {data_dir}')

    train_x_ids, val_x_ids, test_x_ids = get_x_ids(data_dir,
                                                   val_split=val_split,
                                                   test_split=test_split,
                                                   max_n=max_n,
                                                   use_cached=used_cached_x_ids)
    log.info(f'train_x_ids length = {len(train_x_ids)}')
    log.info(f'val_x_ids length = {len(val_x_ids)}')
    log.info(f'test_x_ids length = {len(test_x_ids)}')
    log.info(f'batch_size = {batch_size}')
    # exit()

    train_gen = data_gen(train_x_ids,
                         n_effects,
                         effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                         batch_size=batch_size)
    val_gen = data_gen(val_x_ids,
                       n_effects,
                       effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                       batch_size=batch_size)

    model = architecture(in_x=in_x,
                         in_y=in_y,
                         n_mfcc=n_mfcc,
                         n_channels=n_channels,
                         n_effects=n_effects)
    model.summary()

    if load_prev_model:
        log.info('Loading previous best model.')
        model.load_weights(os.path.join(OUT_DIR, f'{model_name}__best.h5'))

    log.info(f'loss = {loss}')
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metric)

    train_model_gen(model,
                    train_gen,
                    val_gen,
                    model_name,
                    epochs=epochs,
                    patience=patience,
                    min_delta=min_delta,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers)
