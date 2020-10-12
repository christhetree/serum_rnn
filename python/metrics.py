import logging
import os

import librosa as lr
import numpy as np
from sklearn.metrics import mean_squared_error

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def calc_mfcc_metric(mel_a: np.ndarray,
                     mel_b: np.ndarray,
                     n_mfcc: int,
                     sr: int,
                     normalized_mel: bool = True,
                     top_db: float = 80.0) -> float:
    power_mel_a = mel_a
    power_mel_b = mel_b
    if normalized_mel:
        power_mel_a = mel_a * top_db
        power_mel_b = mel_b * top_db

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


def calc_lsd(mel_a: np.ndarray,
             mel_b: np.ndarray,
             normalized_mel: bool = True,
             top_db: float = 80.0) -> float:
    assert mel_a.shape == mel_b.shape
    power_mel_a = mel_a
    power_mel_b = mel_b
    if normalized_mel:
        power_mel_a = mel_a * top_db
        power_mel_b = mel_b * top_db

    spec_dists = []
    for idx in range(mel_a.shape[1]):
        mel_row_a = power_mel_a[:, idx]
        mel_row_b = power_mel_b[:, idx]
        spec_dist = np.sqrt(mean_squared_error(mel_row_a, mel_row_b))
        spec_dists.append(spec_dist)

    mean_dist = np.mean(spec_dists)
    return mean_dist
