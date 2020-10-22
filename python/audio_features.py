import logging
import os
from collections import namedtuple
from typing import Optional

import librosa as lr
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

from config import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

AudioFeatures = namedtuple(
    'AudioFeatures',
    'mel mfcc spec_cent spec_bw spec_bw_hi spec_bw_lo spec_flat'
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
                 sr: int,
                 n_fft: int,
                 hop_length: int,
                 n_mels: int,
                 max_n_of_frames: Optional[int] = None,
                 assert_n_of_frames: bool = True,
                 norm_audio: bool = False,
                 norm_mel: bool = True,
                 fmin: int = 20,
                 fmax: int = 16000,
                 db_ref: float = 1.0,
                 top_db: float = 80.0,
                 n_mfcc: int = 0,
                 calc_cent: bool = False,
                 calc_bw: bool = False,
                 calc_flat: bool = False) -> AudioFeatures:
    assert fmax <= sr / 2
    assert len(audio.shape) == 1
    if calc_bw:
        assert calc_cent

    if norm_audio:
        audio = lr.util.normalize(audio)

    if max_n_of_frames:
        audio_length = audio.shape[0]

        if assert_n_of_frames:
            assert audio_length == max_n_of_frames
        else:
            audio = audio[:max_n_of_frames]

            if audio_length < max_n_of_frames:
                audio = np.concatenate(
                    [audio, np.zeros(max_n_of_frames - audio_length)])

    mel = lr.feature.melspectrogram(y=audio,
                                    sr=sr,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    n_mels=n_mels,
                                    fmin=fmin,
                                    fmax=fmax)
    mel_db = lr.power_to_db(mel, ref=db_ref, top_db=top_db)
    out_mel = mel_db

    if norm_mel:
        out_mel = mel_db / top_db

    mfcc = None
    if n_mfcc:
        mfcc = lr.feature.mfcc(S=mel_db, sr=sr, n_mfcc=n_mfcc)

    spec_cent = None
    out_cent = None
    if calc_cent:
        spec_cent = lr.feature.spectral_centroid(y=audio,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length)
        out_cent = lr.amplitude_to_db(spec_cent,
                                      ref=db_ref,
                                      top_db=top_db)
        if norm_mel:
            out_cent /= top_db

    spec_bw = None
    spec_bw_hi = None
    spec_bw_lo = None
    nyquist_freq = (sr / 2)
    if calc_bw:
        spec_bw = lr.feature.spectral_bandwidth(y=audio,
                                                sr=sr,
                                                centroid=spec_cent)
        half_spec_bw = spec_bw / 2.0
        spec_bw_hi = np.clip(spec_cent + half_spec_bw,
                             a_min=0.0,
                             a_max=nyquist_freq)
        spec_bw_lo = np.clip(spec_cent - half_spec_bw,
                             a_min=0.0,
                             a_max=nyquist_freq)

        if np.min(spec_bw_lo) < 0.0:
            log.warning('spec_bw_lo contains min freq less than 0.0')

        if np.max(spec_bw_hi) > nyquist_freq:
            log.warning('spec_bw_hi contains max freq greater than nyquist')

        spec_bw_hi = lr.amplitude_to_db(spec_bw_hi,
                                        ref=db_ref,
                                        top_db=top_db)
        spec_bw_lo = lr.amplitude_to_db(spec_bw_lo,
                                        ref=db_ref,
                                        top_db=top_db)
        if norm_mel:
            spec_bw /= nyquist_freq
            spec_bw_hi /= top_db
            spec_bw_lo /= top_db

    spec_flat = None
    if calc_flat:
        spec_flat = lr.feature.spectral_flatness(y=audio,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length)

    audio_features = AudioFeatures(mel=out_mel,
                                   mfcc=mfcc,
                                   spec_cent=out_cent,
                                   spec_bw=spec_bw,
                                   spec_bw_hi=spec_bw_hi,
                                   spec_bw_lo=spec_bw_lo,
                                   spec_flat=spec_flat)
    return audio_features


if __name__ == '__main__':
    audio, sr = sf.read(os.path.join(DATA_DIR, 'sample_audio.wav'))
    assert sr == 44100
    features = get_mel_spec(audio,
                            sr=sr,
                            n_fft=2048,
                            hop_length=512,
                            n_mels=128,
                            norm_mel=True,
                            fmin=20,
                            fmax=16000,
                            db_ref=1.0,
                            top_db=80.0,
                            calc_cent=True,
                            calc_bw=True,
                            calc_flat=True)

    exit()
