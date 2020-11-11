import logging
import os
from typing import List, Tuple

import librosa as lr
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

MSSL_N_FFT_S = [64, 128, 256, 512, 1024, 2048]


# TODO: normalization
def mae(db_mel_a: np.ndarray, db_mel_b: np.ndarray) -> float:
    mae_val = mean_absolute_error(db_mel_a, db_mel_b)
    assert 0.0 <= mae_val < 1.0
    return mae_val


# TODO: normalization
def mse(db_mel_a: np.ndarray, db_mel_b: np.ndarray) -> float:
    mse_val = mean_squared_error(db_mel_a, db_mel_b)
    assert 0.0 <= mse_val < 1.0
    return mse_val


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
    assert 0.0 <= mean_dist < 1000.0
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
    assert 0.0 <= mean_dist < 100.0
    return mean_dist


def pcc(db_mel_a: np.ndarray,
        db_mel_b: np.ndarray,
        normalized_mel: bool = True,
        top_db: float = 80.0) -> float:
    assert db_mel_a.shape == db_mel_b.shape
    power_mel_a = db_mel_a
    power_mel_b = db_mel_b
    if normalized_mel:
        power_mel_a = db_mel_a * top_db
        power_mel_b = db_mel_b * top_db

    flat_a = power_mel_a.flatten()
    flat_b = power_mel_b.flatten()

    pcc_val, p_val = pearsonr(flat_a, flat_b)
    return pcc_val


def _col_mae(stft_a: np.ndarray, stft_b: np.ndarray) -> float:
    assert stft_a.shape == stft_b.shape
    aes = []
    for idx in range(stft_a.shape[1]):
        stft_row_a = stft_a[:, idx]
        stft_row_b = stft_b[:, idx]
        ae = np.sum(np.abs(stft_row_a - stft_row_b))
        aes.append(ae)

    mae = np.mean(aes)
    return mae


def mssm(audio_a: np.ndarray,
         audio_b: np.ndarray,
         alpha: float = 0.1,
         n_fft_s: List[int] = MSSL_N_FFT_S,
         overlap: float = 0.75,
         amin: float = 1e-10) -> List[Tuple[str, float]]:
    assert len(audio_a) == len(audio_b)
    l1_s = []
    col_maes = []
    for n_fft in n_fft_s:
        hop_length = int(n_fft * (1.0 - overlap))
        stft_a, _ = lr.spectrum._spectrogram(y=audio_a,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             power=1)
        stft_b, _ = lr.spectrum._spectrogram(y=audio_b,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             power=1)
        log_stft_a = np.log(stft_a + amin)
        log_stft_b = np.log(stft_b + amin)

        l1 = np.linalg.norm(stft_a - stft_b, ord=1) / n_fft
        log_l1 = np.linalg.norm(log_stft_a - log_stft_b, ord=1) / n_fft
        log_l1 *= alpha

        col_mae = _col_mae(stft_a, stft_b) / n_fft
        log_col_mae = _col_mae(log_stft_a, log_stft_b) / n_fft
        log_col_mae *= alpha

        assert col_mae <= l1
        assert log_col_mae <= log_l1
        log.debug(f'n_fft = {n_fft}')
        log.debug(f'l1 = {l1:.5f}')
        log.debug(f'alpha * log_l1 = {log_l1:.5f}')
        log.debug(f'col_mae = {col_mae:.5f}')
        log.debug(f'alpha * log_col_mae = {log_col_mae:.5f}')

        total_l1 = l1 + log_l1
        total_col_mae = col_mae + log_col_mae
        l1_s.append(total_l1)
        col_maes.append(total_col_mae)

    final_l1 = np.mean(l1_s)
    final_col_mae = np.mean(col_maes)
    return [('mssm', final_l1), ('mssmm', final_col_mae)]


if __name__ == '__main__':
    import soundfile as sf
    from config import DATA_DIR
    from audio_features import get_mel_spec

    # herp = np.linalg.norm(np.array([[1, 2], [0, -1]]) - np.array([[3, 4], [0, 2]]), ord=1)
    # derp = np.array([[1, 3, 4],
    #                  [0, 4, 0]])
    # print(derp.shape)
    # herp = np.linalg.norm(derp, ord=1)
    # print(herp)
    # exit()

    audio_b, sr = sf.read(os.path.join(DATA_DIR, 'square__dry.wav'))
    # audio_a, sr = sf.read(os.path.join(DATA_DIR, 'saw__dry.wav'))
    # audio_a, sr = sf.read(os.path.join(DATA_DIR, 'triangle__dry.wav'))
    audio_a, sr = sf.read(os.path.join(DATA_DIR, 'sine__dry.wav'))

    mssm_dist = mssm(audio_a, audio_b)
    # mssm = calc_multi_scale_spectral_loss(audio_b, audio_a)
    print(mssm_dist)
    exit()

    mel_a = get_mel_spec(audio_a, sr, 2048, 512, 128).mel
    mel_b = get_mel_spec(audio_b, sr, 2048, 512, 128).mel

    pcc_val = lsd(mel_a, mel_b, normalized_mel=False)
    print(pcc_val)
    pcc_val = lsd(mel_b, mel_a, normalized_mel=False)
    print(pcc_val)
    pcc_val = lsd(mel_a, mel_a, normalized_mel=False)
    print(pcc_val)
