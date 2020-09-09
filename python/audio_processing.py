import logging
import ntpath
import os
from collections import namedtuple, defaultdict
from typing import Optional, Dict, Union, Set

import librosa as lr
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter
from tqdm import tqdm

from python.config import MEL_SR, HOP_LENGTH, N_MELS, N_FFT, CONFIGS_DIR
from python.effects import get_effect, param_to_type, DESC_TO_PARAM, \
    param_to_effect, PARAM_TO_DESC

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

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

    log.info('Shuffling render names.')
    np.random.shuffle(render_names)

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


def parse_save_name(save_name: str) -> Dict[str, Union[int, float, bool]]:
    save_name = os.path.splitext(save_name.strip())[0]  # Get rid of file ext.
    tokens = save_name.split('__')
    name = tokens[0]
    result = {'name': name}

    for token in tokens[1:]:
        sub_tokens = token.split('_')
        if len(sub_tokens) > 1:
            key = '_'.join(sub_tokens[:-1])
            value = sub_tokens[-1]
            if '.' in value:
                result[key] = float(value)
            elif value == 'T':
                result[key] = True
            elif value == 'F':
                result[key] = False
            else:
                result[key] = int(value)
        else:
            result[token] = True

    return result


def reverse_render_hash() -> Dict[int, Union[int, float]]:
    pass


def generate_y(path: str,
               params: Optional[Set[int]] = None) -> None:
    assert os.path.isfile(path)

    data_npz_name = os.path.splitext(ntpath.basename(path))[0]
    effect_dir_name = os.path.normpath(path).split(os.path.sep)[-3]
    log.info(f'.npz file name: {data_npz_name}')
    log.info(f'Effect dir name: {effect_dir_name}')

    effect_dir_info = parse_save_name(effect_dir_name)
    granularity = effect_dir_info['gran']
    effect_names = effect_dir_info['name'].split('_')
    log.info(f'Using granularity of {granularity}')
    log.info(f'{effect_names} effects found.')

    if params is None:
        log.info('No params provided. Calculating y for all params.')
        params = set()
        for effect_name in effect_names:
            effect = get_effect(effect_name)
            for param in effect.order:
                params.add(param)

    log.info(f'Calculating y for the following params: {sorted(list(params))}')

    param_types = defaultdict(list)
    for param in params:
        param_types[param_to_type[param]].append(param)
    binary_params = sorted(param_types['binary'])
    categorical_params = sorted(param_types['categorical'])
    continuous_params = sorted(param_types['continuous'])

    log.info(f'Binary params: {binary_params}')
    log.info(f'Categorical params: {categorical_params}')
    log.info(f'Continuous params: {continuous_params}')

    data = np.load(path)
    render_names = data['render_names'].tolist()
    log.info(f'{len(render_names)} renders found in {data_npz_name}')
    y = defaultdict(list)

    for render_name in tqdm(render_names):
        render_name_info = parse_save_name(render_name)
        render_name_params = {}
        for desc, value in render_name_info.items():
            if desc == 'name':
                continue

            param = DESC_TO_PARAM[desc]
            render_name_params[param] = value

        bin_values = []
        for param in binary_params:
            if param in render_name_params:
                bin_values.append(float(render_name_params[param]))
            else:
                effect = param_to_effect[param]
                bin_values.append(effect.default[param])

        if bin_values:
            y['binary'].append(bin_values)

        for param in categorical_params:
            desc = PARAM_TO_DESC[param]
            if param in render_name_params:
                value = int(render_name_params[param])
            else:
                effect = param_to_effect[param]
                n_categories = effect.categorical[param]
                value = int((effect.default[param] * n_categories) + 0.5)
            y[desc].append(value)

        cont_values = []
        for param in continuous_params:
            if param in render_name_params:
                cont_values.append(render_name_params[param] / granularity)
            else:
                effect = param_to_effect[param]
                cont_values.append(effect.default[param])

        if cont_values:
            y['continuous'].append(cont_values)

    y = dict(y)

    log.info('Converting to ndarray.')
    for key in tqdm(y):
        if key == 'binary' or key == 'continous':
            y[key] = np.array(y[key], dtype=np.float32)
        else:
            y[key] = np.array(y[key], dtype=np.int32)

        log.info(f'{key} ndarray shape: {y[key].shape}')

    n_categories = []
    param_to_desc = []
    for param in categorical_params:
        effect = param_to_effect[param]
        n_categories = effect.categorical[param]
        desc = PARAM_TO_DESC[param]
        param_to_desc.append(desc)

    log.info(f'n_categories = {n_categories}')
    if binary_params:
        y['binary_params'] = np.array(binary_params, dtype=np.int32)
    if categorical_params:
        y['categorical_params'] = np.array(categorical_params, dtype=np.int32)
        y['n_categories'] = np.array(n_categories, dtype=np.int32)
        y['param_to_desc'] = np.array(param_to_desc)
    if continuous_params:
        y['continuous_params'] = np.array(binary_params, dtype=np.int32)

    save_dir = os.path.split(path)[0]
    save_name = f'{data_npz_name}_y.npz'
    save_path = os.path.join(save_dir, save_name)
    log.info(f'Saving as {save_name}')
    np.savez(save_path, **y)


if __name__ == '__main__':
    # process_audio(os.path.join(CONFIGS_DIR, 'audio_process_test.yaml'))
    generate_y('/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/distortion__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_256__hop_len_256__norm_audio_F__norm_mel_T__n_1414.npz',
               params=None)
