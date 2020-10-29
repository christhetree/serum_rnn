import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from tensorflow.keras import Model

from audio_features import AudioFeatures
from eval_util import FIXED_EFFECT_SEQ
from training_rnn import EFFECT_TO_IDX_MAPPING

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class NextEffectWrapper(ABC):
    def __init__(self, effect_name_to_idx: Dict[str, int]) -> None:
        effect_idx_to_name = {v: k for k, v in effect_name_to_idx.items()}
        self.effect_name_to_idx = effect_name_to_idx
        self.effect_idx_to_name = effect_idx_to_name
        self.effect_names = list(effect_name_to_idx.keys())

    @property
    def n_effects(self) -> int:
        return len(self.effect_names)

    def get_next_effect_pred(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> np.ndarray:
        next_effect_name = self.get_next_effect_name(target_af,
                                                     af_seq,
                                                     effect_name_seq,
                                                     effects_can_repeat,
                                                     t_effect_names)
        pred = np.zeros((self.n_effects,), dtype=np.float32)
        effect_idx = self.effect_name_to_idx[next_effect_name]
        pred[effect_idx] = 1.0
        return pred

    @abstractmethod
    def get_next_effect_name(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str]) -> str:
        return NotImplemented


class OracleWrapper(NextEffectWrapper):
    def __init__(
            self,
            is_fixed_seq: bool,
            fixed_effect_seq: List[str] = FIXED_EFFECT_SEQ,
            effect_name_to_idx: Dict[str, int] = EFFECT_TO_IDX_MAPPING
    ) -> None:
        super().__init__(effect_name_to_idx)
        self.is_fixed_seq = is_fixed_seq
        self.fixed_effect_seq = fixed_effect_seq

    def get_next_effect_name(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str]) -> str:
        assert not effects_can_repeat
        used_effects = set(effect_name_seq)
        assert len(used_effects) < self.n_effects

        target_effects = set(t_effect_names)
        all_effects = list(self.fixed_effect_seq)
        if not self.is_fixed_seq:
            np.random.shuffle(all_effects)

        t_remaining = [e for e in all_effects
                       if e in target_effects and e not in used_effects]
        if t_remaining:
            return t_remaining[0]

        all_remaining = [e for e in all_effects if e not in used_effects]
        return all_remaining[0]


class RandomWrapper(OracleWrapper):
    def get_next_effect_name(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> str:
        assert not effects_can_repeat
        used_effects = set(effect_name_seq)
        assert len(used_effects) < self.n_effects
        all_effects = list(self.fixed_effect_seq)
        if not self.is_fixed_seq:
            np.random.shuffle(all_effects)

        remaining = [e for e in all_effects if e not in used_effects]
        return remaining[0]


class NextEffectRNNWrapper(NextEffectWrapper):
    def __init__(
            self,
            next_effect_model: Model,
            effect_name_to_idx: Dict[str, int] = EFFECT_TO_IDX_MAPPING
    ) -> None:
        super().__init__(effect_name_to_idx)
        self.next_effect_model = next_effect_model

    def _convert_effect_name_seq(self,
                                 effect_name_seq: List[str]) -> np.ndarray:
        effect_seq = np.zeros((1, len(effect_name_seq) + 1, self.n_effects + 1),
                              dtype=np.float32)
        effect_seq[0, 0, -1] = 1.0  # Init effect

        for idx, effect_name in enumerate(effect_name_seq):
            effect_idx = self.effect_name_to_idx[effect_name]
            assert effect_idx != self.n_effects
            effect_seq[0, idx + 1, effect_idx] = 1.0

        return effect_seq

    def _convert_af_seq(self,
                        target_af: AudioFeatures,
                        af_seq: List[AudioFeatures]) -> (np.ndarray,
                                                         np.ndarray):
        mel_seq = [np.stack([target_af.mel, af.mel], axis=-1) for af in af_seq]
        mel_seq = np.array(mel_seq, dtype=np.float32)
        mel_seq = np.expand_dims(mel_seq, axis=0)
        mfcc_seq = [np.stack([target_af.mfcc, af.mfcc], axis=-1)
                    for af in af_seq]
        mfcc_seq = np.array(mfcc_seq, dtype=np.float32)
        mfcc_seq = np.expand_dims(mfcc_seq, axis=0)

        return mel_seq, mfcc_seq

    def get_next_effect_pred(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> np.ndarray:
        mel_seq, mfcc_seq = self._convert_af_seq(target_af, af_seq)
        effect_seq = self._convert_effect_name_seq(effect_name_seq)
        x = [mel_seq, mfcc_seq, effect_seq]
        pred = self.next_effect_model.predict(x, batch_size=1)[0]
        return pred

    def get_next_effect_name(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> str:
        pred = self.get_next_effect_pred(target_af,
                                         af_seq,
                                         effect_name_seq,
                                         effects_can_repeat)
        if effects_can_repeat:
            next_effect_idx = np.argmax(pred)
            next_effect_name = self.effect_idx_to_name[next_effect_idx]
            return next_effect_name

        used_effects = set(effect_name_seq)
        assert len(used_effects) < self.n_effects

        next_effect_name = None
        highest_prob = 0.0
        for idx, prob in enumerate(pred):
            effect_name = self.effect_idx_to_name[idx]
            if effect_name not in used_effects and prob > highest_prob:
                next_effect_name = effect_name
                highest_prob = prob

        return next_effect_name


class NextEffectSeqOnlyRNNWrapper(NextEffectRNNWrapper):
    def get_next_effect_pred(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> np.ndarray:
        mel_seq, mfcc_seq = self._convert_af_seq(target_af, af_seq)
        init_mel = mel_seq[:, 0, :, :, :]
        init_mfcc = mfcc_seq[:, 0, :, :, :]
        effect_seq = self._convert_effect_name_seq(effect_name_seq)
        x = [init_mel, init_mfcc, effect_seq]
        pred = self.next_effect_model.predict(x, batch_size=1)[0]
        return pred


class AllEffectsCNNWrapper(NextEffectRNNWrapper):
    def get_next_effect_pred(self,
                             target_af: AudioFeatures,
                             af_seq: List[AudioFeatures],
                             effect_name_seq: List[str],
                             effects_can_repeat: bool,
                             t_effect_names: List[str] = None) -> str:
        assert not effects_can_repeat
        mel_seq, mfcc_seq = self._convert_af_seq(target_af, af_seq)
        init_mel = mel_seq[:, 0, :, :, :]
        init_mfcc = mfcc_seq[:, 0, :, :, :]
        x = [init_mel, init_mfcc]
        pred = self.next_effect_model.predict(x, batch_size=1)[0]
        return pred
