import logging
import os
from collections import namedtuple
from typing import Optional, Set, List

import librosa as lr
import numpy as np
from scipy.signal import butter, lfilter

from config import MEL_SR, HOP_LENGTH, N_MELS, N_FFT
from effects import get_effect, DESC_TO_PARAM, \
    PARAM_TO_EFFECT, PARAM_TO_DESC
from util import parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

ProcessConfig = namedtuple(
    'ProcessConfig',
    'hop_length max_n_of_frames n_fft n_mels normalize_audio normalize_mel sr '
    'use_hashes root_dir'
)


def bandpass_filter(y: np.ndarray,
                    sr: int,
                    lo_cut: int,
                    hi_cut: int,
                    order: int = 6,
                    normalize_output: bool = False) -> Optional[np.ndarray]:
    nyq = 0.5 * sr
    assert nyq > hi_cut and lo_cut > 0

    lo = lo_cut / nyq
    hi = hi_cut / nyq
    b, a = butter(N=order, Wn=[lo, hi], btype='band')
    filtered_y = lfilter(b, a, y).astype(np.float32)

    if not np.all(np.isfinite(filtered_y)):
        log.info(f'Bandpass filter of [{lo_cut}, {hi_cut}] failed')
        return None

    if normalize_output:
        filtered_y = lr.util.normalize(filtered_y)

    return filtered_y


def add_noise(y: np.ndarray,
              snr: float,
              normalize_output: bool = False) -> np.ndarray:
    audio_rms = np.sqrt(np.mean(y ** 2))
    noise_rms = np.sqrt((audio_rms ** 2) / (10 ** (snr / 10.0)))
    noise_std = noise_rms
    noise = np.random.normal(0, noise_std, y.shape[0]).astype(np.float32)
    output = y + noise

    if normalize_output:
        output = lr.util.normalize(output)

    return output


def get_mel_spec(audio: np.ndarray,
                 sr: int = MEL_SR,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 max_n_of_frames: Optional[int] = None,
                 normalize_audio: bool = True,
                 normalize_mel: bool = True,
                 db_ref: float = 1.0,
                 top_db: float = 80.0) -> np.ndarray:
    if normalize_audio:
        audio = lr.util.normalize(audio)

    if max_n_of_frames:
        audio = audio[:max_n_of_frames]
        audio_length = audio.shape[0]

        if audio_length < max_n_of_frames:
            audio = np.concatenate(
                [audio, np.zeros(max_n_of_frames - audio_length)])

    mel_spec = lr.feature.melspectrogram(
        audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = lr.power_to_db(mel_spec, ref=db_ref, top_db=top_db)

    if normalize_mel:
        mel_spec = mel_spec / top_db

    return mel_spec


def create_save_dir(pc: ProcessConfig,
                    create_dirs: bool = True) -> str:
    assert os.path.exists(pc.root_dir)
    save_dir_name = create_save_name(pc)

    save_dir = os.path.join(pc.root_dir, 'processing')
    if not create_dirs:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        log.info('Creating processing folder.')
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, save_dir_name)
    if not create_dirs:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        log.info(f'Creating dir: {save_dir_name}')
        os.makedirs(save_dir)

    return save_dir


def create_save_name(pc: ProcessConfig) -> str:
    save_name = f'mel__sr_{pc.sr}__frames_{pc.max_n_of_frames}__' \
                f'n_fft_{pc.n_fft}__n_mels_{pc.n_mels}__' \
                f'hop_len_{pc.hop_length}__' \
                f'norm_audio_{str(pc.normalize_audio)[0]}__' \
                f'norm_mel_{str(pc.normalize_mel)[0]}'
    return save_name


def generate_base_render_hash(orig_render_name: str,
                              base_effect_names: List[str],
                              exclude_descs: Set[str]) -> str:
    hash_tokens = ['_'.join(base_effect_names)]

    for param_str in orig_render_name.split('__'):
        if param_str.endswith('.wav'):
            param_str = param_str[:-4]
        split_param = param_str.split('_')
        desc = '_'.join(split_param[:-1])
        if desc in DESC_TO_PARAM:
            param = DESC_TO_PARAM[desc]
            effect = PARAM_TO_EFFECT[param]
            if effect.name in base_effect_names and desc not in exclude_descs:
                hash_tokens.append(param_str)

    render_hash = '__'.join(hash_tokens)

    if not render_hash:
        render_hash = 'dry'

    render_hash = f'{render_hash}.wav'
    return render_hash


def get_base_effect_info(orig_effect_dir_name: str,
                         exclude_descs: Set[str] = None) -> (str, List[str]):
    if exclude_descs is None:
        exclude_descs = set()

    orig_effect_dir_info = parse_save_name(orig_effect_dir_name, is_dir=True)
    gran = orig_effect_dir_info['gran']
    orig_effect_names = orig_effect_dir_info['name'].split('_')

    base_effect_names = []
    for effect_name in orig_effect_names:
        effect = get_effect(effect_name)
        if not all(PARAM_TO_DESC[p] in exclude_descs for p in effect.order):
            base_effect_names.append(effect_name)
    if not base_effect_names:
        base_effect_names.append('dry')

    base_effect_names = sorted(list(set(base_effect_names)))
    base_effect_dir_name = f'{"_".join(base_effect_names)}__gran_{gran}'
    return base_effect_dir_name, base_effect_names
