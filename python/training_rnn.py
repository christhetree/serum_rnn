import logging
import os
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.data_utils import Sequence
from tqdm import tqdm

from config import OUT_DIR, DATASETS_DIR
from models import baseline_cnn_2x, baseline_effect_rnn, baseline_cnn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

GPU = 0
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log.info(f'GPUs available: {physical_devices}')
    tf.config.experimental.set_visible_devices(physical_devices[GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[GPU], enable=True)


# EFFECT_TO_IDX_MAPPING = {
#     'compressor': 0,
#     'distortion': 1,
#     'eq': 2,
#     'flanger': 3,
#     'phaser': 4,
# }
EFFECT_TO_IDX_MAPPING = {
    'compressor': 0,
    'distortion': 1,
    'eq': 2,
    'phaser': 3,
    'reverb-hall': 4
}


class RNNDataGenerator(Sequence):
    def __init__(self,
                 x_ids: List[Tuple[List[str], List[str]]],
                 in_x: int,
                 in_y: int,
                 n_effects: int,
                 effect_name_to_idx: Dict[str, int],
                 batch_size: int = 128,
                 shuffle: bool = True) -> None:
        assert len(x_ids) >= batch_size

        if shuffle:
            np.random.shuffle(x_ids)

        self.x_ids = x_ids
        self.in_x = in_x
        self.in_y = in_y
        self.n_effects = n_effects
        self.effect_name_to_idx = effect_name_to_idx
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return int(np.floor(len(self.x_ids) / self.batch_size))

    def __getitem__(self, idx: int) -> ((np.ndarray, np.ndarray), np.ndarray):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_x_ids = self.x_ids[start_idx:end_idx]
        x = self._create_x_batch(batch_x_ids)
        y = self._create_y_batch(batch_x_ids)
        return x, y

    def _create_x_batch(
            self,
            batch_x_ids: List[Tuple[List[str], List[str]]]
    ) -> (np.ndarray, np.ndarray):
        mel_seqs = []
        effect_seqs = []
        for mel_path_seq, effect_names in batch_x_ids:
            target_mel = np.load(mel_path_seq[-1])['mel']
            mel_seq = []
            for mel_path in mel_path_seq[:-1]:
                mel = np.load(mel_path)['mel']
                mel_seq.append(np.stack([target_mel, mel], axis=-1))
            mel_seqs.append(mel_seq)

            effect_seq = []
            for effect_name in effect_names[:-1]:
                one_hot = np.zeros((self.n_effects + 1,), dtype=np.float32)
                effect_idx = self.effect_name_to_idx.get(effect_name, -1)
                one_hot[effect_idx] = 1.0
                effect_seq.append(one_hot)
            effect_seqs.append(effect_seq)

        padded_mel_seqs = pad_sequences(mel_seqs,
                                        value=0.0,
                                        padding='post',
                                        dtype='float32')
        padded_effect_seqs = pad_sequences(effect_seqs,
                                           value=0.0,
                                           padding='post',
                                           dtype='float32')
        return padded_mel_seqs, padded_effect_seqs

    def _create_y_batch(
            self, batch_x_ids: List[Tuple[List[str], List[str]]]) -> np.ndarray:
        y = np.array(
            [self.effect_name_to_idx[effect_names[-1]]
             for _, effect_names in batch_x_ids],
            dtype=np.int32
        )
        return y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.x_ids)


def train_model_gen(model: Model,
                    train_gen: RNNDataGenerator,
                    val_gen: RNNDataGenerator,
                    model_name: str,
                    epochs: int = 100,
                    patience: int = 10,
                    output_dir_path: str = OUT_DIR,
                    use_multiprocessing: bool = True,
                    workers: int = 8) -> None:
    save_path = os.path.join(
        output_dir_path,
        # model_name + '_e{epoch:03d}_vl{val_loss:.4f}.h5'
        f'{model_name}__best.h5'
    )
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
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
    in_x = 128
    in_y = 88
    n_channels = 2
    n_effects = len(EFFECT_TO_IDX_MAPPING)

    cnn_architecture = baseline_cnn
    # cnn_architecture = baseline_cnn_2x
    # cnn_architecture = exposure_cnn
    # cnn_architecture = baseline_lstm

    batch_size = 32
    epochs = 100
    val_split = 0.10
    test_split = 0.05
    patience = 10
    used_cached_x_ids = True
    max_n = -1
    # use_multiprocessing = False
    use_multiprocessing = True
    workers = 8
    load_prev_model = False

    presets_cat = 'basic_shapes'
    # presets_cat = 'adv_shapes'
    # presets_cat = 'temporal'

    # model_name = f'testing__rnn'
    model_name = f'seq_5_v3__{presets_cat}__rnn__{cnn_architecture.__name__}'

    datasets_dir = DATASETS_DIR
    # data_dir = os.path.join(datasets_dir, f'testing__rnn')
    data_dir = os.path.join(datasets_dir, f'seq_5_v3__{presets_cat}__rnn')
    log.info(f'data_dir = {data_dir}')

    train_x_ids, val_x_ids, test_x_ids = get_x_ids(data_dir,
                                                   val_split=val_split,
                                                   test_split=test_split,
                                                   max_n=max_n)
    log.info(f'train_x_ids length = {len(train_x_ids)}')
    log.info(f'val_x_ids length = {len(val_x_ids)}')
    log.info(f'test_x_ids length = {len(test_x_ids)}')
    log.info(f'batch_size = {batch_size}')

    train_gen = RNNDataGenerator(train_x_ids,
                                 in_x,
                                 in_y,
                                 n_effects,
                                 effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                                 batch_size=batch_size)
    val_gen = RNNDataGenerator(val_x_ids,
                               in_x,
                               in_y,
                               n_effects,
                               effect_name_to_idx=EFFECT_TO_IDX_MAPPING,
                               batch_size=batch_size)

    model = baseline_effect_rnn(in_x,
                                in_y,
                                n_channels,
                                n_effects,
                                cnn_architecture=cnn_architecture)

    if load_prev_model:
        log.info('Loading previous best model.')
        model.load_weights(os.path.join(OUT_DIR, f'{model_name}__best.h5'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics='acc')
    model.summary()

    train_model_gen(model,
                    train_gen,
                    val_gen,
                    model_name,
                    epochs=epochs,
                    patience=patience,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers)
