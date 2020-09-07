import logging
import os
from collections import namedtuple
from typing import Optional

import librosa as lr
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter
from tqdm import tqdm

from python.config import MEL_SR, HOP_LENGTH, N_MELS, N_FFT, MEL_MAX_DUR, \
    DATASETS_DIR, RM_SR, CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'DEBUG'))

ProcessConfig = namedtuple(
    'ProcessConfig',
    'hop_length max_n_of_frames n_fft n_mels normalize_audio normalize_mel sr '
    'root_dir'
)


def get_mel_spec(audio: np.ndarray,
                 sr: int = MEL_SR,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 max_n_of_frames: Optional[int] = None,
                 normalize_audio: bool = True,
                 normalize_mel: bool = True) -> np.ndarray:
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


def process_audio(process_config_path: str) -> None:
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)

    pc = ProcessConfig(**process_config)
    assert os.path.exists(pc.root_dir)

    save_dir = os.path.join(pc.root_dir, 'processing')
    if not os.path.exists(save_dir):
        log.info('Creating processing folder.')
        os.makedirs(save_dir)

    save_name = f'mel__sr_{pc.sr}__frames_{pc.max_n_of_frames}__' \
                f'n_fft_{pc.n_fft}__n_mels_{pc.n_mels}__' \
                f'hop_len_{pc.hop_length}__' \
                f'norm_audio_{str(pc.normalize_audio)[0]}__' \
                f'norm_mel_{str(pc.normalize_mel)[0]}'

    npz_names = []
    for npz_name in os.listdir(save_dir):
        if npz_name.startswith(save_name):
            npz_names.append(npz_name)

    assert len(npz_names) < 2
    existing_npz_name = None
    if npz_names:
        existing_npz_name = npz_names[0]
        proc_data = np.load(os.path.join(save_dir, existing_npz_name))
        proc_render_names = proc_data['render_names'].tolist()
        log.info('Loading existing mels.')
        mels = proc_data['mels']
        log.info(f'Found {len(proc_render_names)} existing processed renders '
                 f'in {existing_npz_name}')
    else:
        proc_render_names = []
        mels = []

    proc_hashes = set(proc_render_names)
    assert len(mels) == len(proc_render_names)
    assert len(proc_hashes) == len(proc_render_names)

    render_names = []
    for render_name in os.listdir(pc.root_dir):
        if render_name.endswith('.wav'):
            render_names.append(render_name)
    log.info(f'{len(render_names)} renders found in {pc.root_dir}')

    new_proc_render_names = []
    new_mels = []
    for render_name in tqdm(render_names):
        if render_name in proc_hashes:
            log.debug(f'{render_name} has already been processed.')
            continue

        audio_path = os.path.join(pc.root_dir, render_name)
        audio, sr = sf.read(audio_path)

        assert sr == pc.sr
        if len(audio) != pc.max_n_of_frames:
            log.warning(f'{render_name} has incorrect length: {len(audio)}')

        mel = get_mel_spec(audio,
                           sr=pc.sr,
                           hop_length=pc.hop_length,
                           n_mels=pc.n_mels,
                           n_fft=pc.n_fft,
                           max_n_of_frames=pc.max_n_of_frames,
                           normalize_audio=pc.normalize_audio,
                           normalize_mel=pc.normalize_mel)
        new_mels.append(mel)
        new_proc_render_names.append(render_name)
        proc_hashes.add(render_name)

    assert len(new_mels) == len(new_proc_render_names)
    log.info(f'{len(new_mels)} renders processed.')

    if len(new_mels) == 0:
        return

    log.info('Converting mels to ndarray.')
    new_mels = np.array(new_mels, dtype=np.float32)

    if existing_npz_name:
        log.info(f'Prev. mels shape = {mels.shape}')
        proc_render_names.extend(new_proc_render_names)
        log.info('Concatenating mels.')
        mels = np.concatenate([mels, new_mels], axis=0)
    else:
        proc_render_names = new_proc_render_names
        mels = new_mels

    log.info(f'Total mels shape = {mels.shape}')
    new_npz_name = f'{save_name}__n_{len(proc_render_names)}.npz'

    log.info(f'Saving new npz file: {new_npz_name}')
    np.savez(os.path.join(save_dir, new_npz_name),
             render_names=proc_render_names,
             mels=mels)

    if existing_npz_name:
        log.info(f'Deleting prev npz file: {existing_npz_name}')
        os.remove(os.path.join(save_dir, existing_npz_name))


if __name__ == '__main__':
    process_audio(os.path.join(CONFIGS_DIR, 'audio_process_test.yaml'))
