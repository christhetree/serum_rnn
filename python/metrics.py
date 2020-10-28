import logging
import os
from typing import List

import librosa as lr
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

MSSL_N_FFT_S = [64, 128, 256, 512, 1024, 2048]


def mae(db_mel_a: np.ndarray, db_mel_b: np.ndarray) -> float:
    return mean_absolute_error(db_mel_a, db_mel_b)


def mse(db_mel_a: np.ndarray, db_mel_b: np.ndarray) -> float:
    return mean_squared_error(db_mel_a, db_mel_b)


def mfcc_dist(db_mel_a: np.ndarray,
              db_mel_b: np.ndarray,
              n_mfcc: int = 30,
              sr: int = 44100,
              normalized_mel: bool = True,
              top_db: float = 80.0) -> float:
    power_mel_a = db_mel_a
    power_mel_b = db_mel_b
    if normalized_mel:
        power_mel_a = db_mel_a * top_db
        power_mel_b = db_mel_b * top_db

    mfcc_a = lr.feature.mfcc(S=power_mel_a, sr=sr, n_mfcc=n_mfcc)
    mfcc_b = lr.feature.mfcc(S=power_mel_b, sr=sr, n_mfcc=n_mfcc)
    assert mfcc_a.shape == mfcc_b.shape

    euc_dists = []
    for idx in range(mfcc_a.shape[1]):
        mfcc_row_a = mfcc_a[:, idx]
        mfcc_row_b = mfcc_b[:, idx]
        euc_dist = np.linalg.norm(mfcc_row_a - mfcc_row_b)
        euc_dists.append(euc_dist)

    mean_dist = np.mean(euc_dists)
    return mean_dist


def lsd(db_mel_a: np.ndarray,
        db_mel_b: np.ndarray,
        normalized_mel: bool = True,
        top_db: float = 80.0) -> float:
    assert db_mel_a.shape == db_mel_b.shape
    power_mel_a = db_mel_a
    power_mel_b = db_mel_b
    if normalized_mel:
        power_mel_a = db_mel_a * top_db
        power_mel_b = db_mel_b * top_db

    spec_dists = []
    for idx in range(db_mel_a.shape[1]):
        mel_row_a = power_mel_a[:, idx]
        mel_row_b = power_mel_b[:, idx]
        spec_dist = np.sqrt(mean_squared_error(mel_row_a, mel_row_b))
        spec_dists.append(spec_dist)

    mean_dist = np.mean(spec_dists)
    return mean_dist


def calc_multi_scale_spectral_loss(audio_a: np.ndarray,
                                   audio_b: np.ndarray,
                                   alpha: float = 1.0,
                                   n_fft_s: List[int] = MSSL_N_FFT_S,
                                   overlap: float = 0.75,
                                   amin: float = 1e-10) -> float:
    assert len(audio_a) == len(audio_b)
    result = 0.0
    for n_fft in n_fft_s:
        hop_length = int(n_fft * (1.0 - overlap))
        stft_a, _ = lr.spectrum._spectrogram(y=audio_a,
                                             n_fft=n_fft,
                                             hop_length=hop_length)
        stft_b, _ = lr.spectrum._spectrogram(y=audio_b,
                                             n_fft=n_fft,
                                             hop_length=hop_length)
        log_stft_a = np.log(stft_a + amin)
        log_stft_b = np.log(stft_b + amin)

        l1 = np.linalg.norm(stft_a - stft_b, ord=1)
        log_l1 = np.linalg.norm(log_stft_a - log_stft_b, ord=1)
        total = l1 + (alpha * log_l1)
        result += total

    return result


if __name__ == '__main__':
    import soundfile as sf
    from config import DATA_DIR

    a = np.zeros((4,))
    np.log(a + 1e-10)

    audio_a, sr = sf.read(os.path.join(DATA_DIR, 'sine__dry.wav'))
    audio_b, sr = sf.read(os.path.join(DATA_DIR, 'square__dry.wav'))
    mssl = calc_multi_scale_spectral_loss(audio_b, audio_a)
    print(mssl)
