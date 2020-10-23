import logging
import os
from collections import namedtuple
from typing import List, Tuple, Any, Dict, Union

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.data_utils import Sequence

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

XYMetaData = namedtuple(
    'XYMetaData',
    'data_dir x_dir in_x in_y n_mfcc y_dir y_params y_params_str n_bin n_cate '
    'n_cont descs cate_names y_losses metrics'
)

EFFECT_TO_IDX_MAPPING = {
    'compressor': 0,
    'distortion': 1,
    'eq': 2,
    'phaser': 3,
    'reverb-hall': 4,
}

EFFECT_TO_Y_PARAMS = {
    'compressor': {270, 271, 272},
    'distortion': {97, 99},
    'eq': {89, 91, 93},
    'phaser': {112, 113, 114},
    'reverb-hall': {81, 84, 86},
}

assert EFFECT_TO_IDX_MAPPING.keys() == EFFECT_TO_Y_PARAMS.keys()


class TestDataGenerator(Sequence):
    def __init__(self,
                 x_ids: List[Tuple[str, str, str]],
                 x_y_metadata: XYMetaData,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 channel_mode: int = 1) -> None:
        assert len(x_ids) >= batch_size
        assert channel_mode == 1
        assert batch_size == 1

        if shuffle:
            log.info('Shuffling x_ids!')
            np.random.shuffle(x_ids)

        assert channel_mode == -1 or channel_mode == 0 or channel_mode == 1
        if channel_mode == -1:
            log.warning('Data generator is using (b, b) channel mode!')
        elif channel_mode == 0:
            log.info('Data generator is using (x, x) channel mode.')
        else:
            log.info('Data generator is using (x, b) channel mode.')

        self.x_ids = x_ids
        self.x_dir = x_y_metadata.x_dir
        self.in_x = x_y_metadata.in_x
        self.in_y = x_y_metadata.in_y
        self.y_dir = x_y_metadata.y_dir
        self.y_params_str = x_y_metadata.y_params_str
        self.n_bin = x_y_metadata.n_bin
        self.descs = x_y_metadata.descs
        self.n_cont = x_y_metadata.n_cont
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.channel_mode = channel_mode

    def __len__(self) -> int:
        return int(np.floor(len(self.x_ids) / self.batch_size))

    def __getitem__(self, idx: int) -> ((np.ndarray, str, str, str), List[np.ndarray]):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_x_ids = self.x_ids[start_idx:end_idx]
        x = self._create_x_batch(batch_x_ids)
        y = self._create_y_batch(batch_x_ids)
        return x, y

    def _create_x_batch(self,
                        batch_x_ids: List[Tuple[str, str, str]]) -> Any:
        x = np.empty((self.batch_size, self.in_x, self.in_y, 2),
                     dtype=np.float32)
        render_name = None
        base_render_name = None
        preset = None

        for idx, (x_id, mel_path, base_mel_path) in enumerate(batch_x_ids):
            if self.channel_mode == 1:
                mel_info = np.load(mel_path)
                base_mel_info = np.load(base_mel_path)
                render_name = mel_info['render_name'].item()
                base_render_name = base_mel_info['render_name'].item()
                preset = x_id.split('__')[0]
                mel = mel_info['mel']
                base_mel = base_mel_info['mel']
                x[idx, :, :, 0] = mel
                x[idx, :, :, 1] = base_mel
            elif self.channel_mode == 0:
                mel = np.load(mel_path)['mel']
                x[idx, :, :, 0] = mel
                x[idx, :, :, 1] = mel
            else:
                base_mel = np.load(base_mel_path)['mel']
                x[idx, :, :, 0] = base_mel
                x[idx, :, :, 1] = base_mel

        return x, render_name, base_render_name, preset

    def _create_y_batch(
            self, batch_x_ids: List[Tuple[str, str, str]]) -> List[np.ndarray]:
        y_bin = None
        y_cates = []
        y_cont = None
        if self.n_bin:
            y_bin = np.empty((self.batch_size, self.n_bin), dtype=np.float32)

        for _ in self.descs:
            y_cates.append(np.empty((self.batch_size,), dtype=np.int32))

        if self.n_cont:
            y_cont = np.empty((self.batch_size, self.n_cont), dtype=np.float32)

        for idx, (x_id, _, _) in enumerate(batch_x_ids):
            y_id = f'{x_id}__y_{self.y_params_str}.npz'
            y_data = np.load(os.path.join(self.y_dir, y_id))
            if self.n_bin:
                y_bin[idx] = y_data['binary']

            for desc, y_cate in zip(self.descs, y_cates):
                y_cate[idx] = y_data[desc]

            if self.n_cont:
                y_cont[idx] = y_data['continuous']

        y = []
        if self.n_bin:
            y.append(y_bin)
        y.extend(y_cates)
        if self.n_cont:
            y.append(y_cont)

        return y


class DataGenerator(Sequence):
    def __init__(self,
                 x_ids: List[Tuple[str, str, str]],
                 x_y_metadata: XYMetaData,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 channel_mode: int = 1) -> None:
        assert len(x_ids) >= batch_size

        if shuffle:
            np.random.shuffle(x_ids)

        assert channel_mode == -1 or channel_mode == 0 or channel_mode == 1
        if channel_mode == -1:
            log.warning('Data generator is using (b, b) channel mode!')
        elif channel_mode == 0:
            log.info('Data generator is using (x, x) channel mode.')
        else:
            log.info('Data generator is using (x, b) channel mode.')

        self.x_ids = x_ids
        self.x_dir = x_y_metadata.x_dir
        self.in_x = x_y_metadata.in_x
        self.in_y = x_y_metadata.in_y
        self.n_mfcc = x_y_metadata.n_mfcc
        self.y_dir = x_y_metadata.y_dir
        self.y_params_str = x_y_metadata.y_params_str
        self.n_bin = x_y_metadata.n_bin
        self.descs = x_y_metadata.descs
        self.n_cont = x_y_metadata.n_cont
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.channel_mode = channel_mode
        self.y_id_to_y_data = {}

    def __len__(self) -> int:
        return int(np.floor(len(self.x_ids) / self.batch_size))

    def __getitem__(self, idx: int) -> (Union[np.ndarray, List[np.ndarray]],
                                        List[np.ndarray]):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_x_ids = self.x_ids[start_idx:end_idx]
        x = self._create_x_batch(batch_x_ids)
        y = self._create_y_batch(batch_x_ids)
        return x, y

    def _create_x_batch(
            self,
            batch_x_ids: List[Tuple[str, ...]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        mel_x = np.empty((self.batch_size, self.in_x, self.in_y, 2),
                          dtype=np.float32)
        mfcc_x = None
        if self.n_mfcc:
            mfcc_x = np.empty((self.batch_size, self.n_mfcc, self.in_y, 2),
                               dtype=np.float32)

        for idx, (_, mel_path, base_mel_path) in enumerate(batch_x_ids):
            if self.channel_mode == 1:
                proc_data = np.load(mel_path)
                mel = proc_data['mel']
                base_proc_data = np.load(base_mel_path)
                base_mel = base_proc_data['mel']
                mel_x[idx, :, :, 0] = mel
                mel_x[idx, :, :, 1] = base_mel

                if self.n_mfcc:
                    mfcc = proc_data['mfcc']
                    base_mfcc = base_proc_data['mfcc']
                    mfcc_x[idx, :, :, 0] = mfcc
                    mfcc_x[idx, :, :, 1] = base_mfcc
            elif self.channel_mode == 0:
                proc_data = np.load(mel_path)
                mel = proc_data['mel']
                mel_x[idx, :, :, 0] = mel
                mel_x[idx, :, :, 1] = mel

                if self.n_mfcc:
                    mfcc = proc_data['mfcc']
                    mfcc_x[idx, :, :, 0] = mfcc
                    mfcc_x[idx, :, :, 1] = mfcc
            else:
                base_proc_data = np.load(base_mel_path)
                base_mel = base_proc_data['mel']
                mel_x[idx, :, :, 0] = base_mel
                mel_x[idx, :, :, 1] = base_mel

                if self.n_mfcc:
                    base_mfcc = base_proc_data['mfcc']
                    mfcc_x[idx, :, :, 0] = base_mfcc
                    mfcc_x[idx, :, :, 1] = base_mfcc

        if self.n_mfcc:
            return [mel_x, mfcc_x]
        else:
            return mel_x

    def _create_y_batch(
            self, batch_x_ids: List[Tuple[str, ...]]) -> List[np.ndarray]:
        y_bin = None
        y_cates = []
        y_cont = None
        if self.n_bin:
            y_bin = np.empty((self.batch_size, self.n_bin), dtype=np.float32)

        for _ in self.descs:
            y_cates.append(np.empty((self.batch_size,), dtype=np.int32))

        if self.n_cont:
            y_cont = np.empty((self.batch_size, self.n_cont), dtype=np.float32)

        for idx, (x_id, _, _) in enumerate(batch_x_ids):
            y_id = f'{x_id}__y_{self.y_params_str}.npz'

            if y_id in self.y_id_to_y_data:
                y_data = self.y_id_to_y_data[y_id]
            else:
                with np.load(os.path.join(self.y_dir, y_id)) as npz_data:
                    y_data = {k: v.copy() for k, v in npz_data.items()}
                self.y_id_to_y_data[y_id] = y_data

            if self.n_bin:
                y_bin[idx] = y_data['binary']

            for desc, y_cate in zip(self.descs, y_cates):
                y_cate[idx] = y_data[desc]

            if self.n_cont:
                y_cont[idx] = y_data['continuous']

        y = []
        if self.n_bin:
            y.append(y_bin)
        y.extend(y_cates)
        if self.n_cont:
            y.append(y_cont)

        return y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.x_ids)


class FastDataGenerator(DataGenerator):
    def _create_x_batch(
            self,
            batch_x_ids: List[Tuple[str, ...]]
    ) -> List[np.ndarray]:
        mels = []
        base_mels = []
        mfccs = []
        base_mfccs = []

        for idx, (_, mel_path, base_mel_path) in enumerate(batch_x_ids):
            if self.channel_mode == 1:
                proc_data = np.load(mel_path)
                mel = proc_data['mel']
                base_proc_data = np.load(base_mel_path)
                base_mel = base_proc_data['mel']
                mels.append(mel)
                base_mels.append(base_mel)

                if self.n_mfcc:
                    mfcc = proc_data['mfcc']
                    base_mfcc = base_proc_data['mfcc']
                    mfccs.append(mfcc)
                    base_mfccs.append(base_mfcc)
            elif self.channel_mode == 0:
                proc_data = np.load(mel_path)
                mel = proc_data['mel']
                mels.append(mel)
                base_mels.append(mel)

                if self.n_mfcc:
                    mfcc = proc_data['mfcc']
                    mfccs.append(mfcc)
                    base_mfccs.append(mfcc)
            else:
                base_proc_data = np.load(base_mel_path)
                base_mel = base_proc_data['mel']
                mels.append(base_mel)
                base_mels.append(base_mel)

                if self.n_mfcc:
                    base_mfcc = base_proc_data['mfcc']
                    mfccs.append(base_mfcc)
                    base_mfccs.append(base_mfcc)

        mels = np.array(mels, dtype=np.float32)
        base_mels = np.array(base_mels, dtype=np.float32)
        mfccs = np.array(mfccs, dtype=np.float32)
        base_mfccs = np.array(base_mfccs, dtype=np.float32)

        return [mels, base_mels, mfccs, base_mfccs]


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
