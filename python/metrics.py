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
    assert 0.0 <= mae_val < 1.0, f'mae_val = {mae_val:.5f}'
    return mae_val


# TODO: normalization
def mse(db_mel_a: np.ndarray, db_mel_b: np.ndarray) -> float:
    mse_val = mean_squared_error(db_mel_a, db_mel_b)
    assert 0.0 <= mse_val < 1.0, f'mse_val = {mse_val:.5f}'
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
    assert 0.0 <= mean_dist < 1000.0, f'mean_dist = {mean_dist:.5f}'
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
    assert 0.0 <= mean_dist < 100.0, f'mean_dist = {mean_dist:.5f}'
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
    assert -1.0 <= pcc_val <= 1.0, f'pcc_val = {pcc_val:.5f}'
    return pcc_val


def mssmae(audio_a: np.ndarray,
           audio_b: np.ndarray,
           alpha: float = 0.1,
           n_fft_s: List[int] = MSSL_N_FFT_S,
           overlap: float = 0.75,
           amin: float = 1e-10) -> List[Tuple[str, float]]:
    assert len(audio_a) == len(audio_b)
    mae_s = []
    mmae_s = []
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

        mae_val = mean_absolute_error(stft_a, stft_b)
        log_mae_val = mean_absolute_error(log_stft_a, log_stft_b)
        log_mae_val *= alpha

        mmae_val = np.linalg.norm(stft_a - stft_b, ord=1) / stft_a.shape[0]
        log_mmae_val = np.linalg.norm(log_stft_a - log_stft_b,
                                      ord=1) / log_stft_a.shape[0]
        log_mmae_val *= alpha

        assert mae_val <= mmae_val
        assert log_mae_val <= log_mmae_val
        log.debug(f'n_fft = {n_fft}')
        log.debug(f'mae_val = {mae_val:.5f}')
        log.debug(f'alpha * log_mae_val = {log_mae_val:.5f}')
        log.debug(f'mmae_val = {mmae_val:.5f}')
        log.debug(f'alpha * log_mmae_val = {log_mmae_val:.5f}')

        total_mae = mae_val + log_mae_val
        total_mmae = mmae_val + log_mmae_val
        mae_s.append(total_mae)
        mmae_s.append(total_mmae)

    final_mae = np.mean(mae_s)
    final_mmae = np.mean(mmae_s)
    assert 0.0 <= final_mae < 50.0, f'final_mae = {final_mae:.5f}'
    assert 0.0 <= final_mmae < 50.0, f'final_mmae = {final_mmae:.5f}'
    return [('mssmae', final_mae), ('mssmmae', final_mmae)]


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
    audio_a, sr = sf.read(os.path.join(DATA_DIR, 'saw__dry.wav'))
    # audio_a, sr = sf.read(os.path.join(DATA_DIR, 'triangle__dry.wav'))
    # audio_a, sr = sf.read(os.path.join(DATA_DIR, 'sine__dry.wav'))

    mssm_dist = mssmae(audio_a, audio_b)
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

# [('mssmae', 0.28394042369110567), ('mssmmae', 0.4049159787337249)]
# [('mssmae', 0.9805862998790783), ('mssmmae', 1.4381192780476386)]  [('mssmae', 0.8020618748511289), ('mssmmae', 1.2854447437727372)]
# [('mssmae', 1.1430242521949447), ('mssmmae', 1.572655287475264)]   [('mssmae', 1.0204597190226088), ('mssmmae', 1.4866313269107636)]  [('mssmae', 0.6536914115660166), ('mssmmae', 1.3568074590235975)]
