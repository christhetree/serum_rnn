import logging
import os
from typing import Optional

import librosa as lr
import numpy as np
from scipy.signal import butter, lfilter

from python.config import MEL_SR, HOP_LENGTH, N_MELS, N_FFT, MEL_MAX_DUR

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


def get_mel_spec(audio: np.ndarray,
                 sr: int = MEL_SR,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 max_len_samples: int = None,
                 max_dur: float = MEL_MAX_DUR,
                 normalize_audio: bool = True,
                 normalize_mel: bool = True) -> np.ndarray:
    if max_len_samples is None:
        max_len_samples = int(max_dur * sr)

    if normalize_audio:
        audio = lr.util.normalize(audio)

    audio = audio[:max_len_samples]
    audio_length = audio.shape[0]
    if audio_length < max_len_samples:
        audio = np.concatenate(
            [audio, np.zeros(max_len_samples - audio_length)])

    mel_spec = lr.feature.melspectrogram(
        audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = lr.power_to_db(mel_spec, np.max)

    if normalize_mel:
        # Axis must be None to normalize across both dimensions at once
        mel_spec = lr.util.normalize(mel_spec, axis=None)

    return mel_spec


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
