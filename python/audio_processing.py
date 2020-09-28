import logging
import ntpath
import os
from collections import namedtuple
from typing import Optional, Set, List

import librosa as lr
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter
from tqdm import tqdm

from config import MEL_SR, HOP_LENGTH, N_MELS, N_FFT, CONFIGS_DIR
from effects import get_effect, DESC_TO_PARAM, \
    param_to_effect, PARAM_TO_DESC
from util import get_render_names, get_mapping, generate_exclude_descs, \
    parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

ProcessConfig = namedtuple(
    'ProcessConfig',
    'hop_length max_n_of_frames n_fft n_mels normalize_audio normalize_mel sr '
    'use_hashes root_dir'
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
    mel_spec = lr.power_to_db(mel_spec, ref=1.0)

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


def _create_save_dir(pc: ProcessConfig,
                     create_dirs: bool = True) -> str:
    assert os.path.exists(pc.root_dir)

    save_dir = os.path.join(pc.root_dir, 'processing')
    if not create_dirs:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        log.info('Creating processing folder.')
        os.makedirs(save_dir)

    return save_dir


def _create_save_name(pc: ProcessConfig) -> str:
    save_name = f'mel__sr_{pc.sr}__frames_{pc.max_n_of_frames}__' \
                f'n_fft_{pc.n_fft}__n_mels_{pc.n_mels}__' \
                f'hop_len_{pc.hop_length}__' \
                f'norm_audio_{str(pc.normalize_audio)[0]}__' \
                f'norm_mel_{str(pc.normalize_mel)[0]}'
    return save_name


def process_audio(process_config_path: str) -> None:
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)

    pc = ProcessConfig(**process_config)
    save_dir = _create_save_dir(pc, create_dirs=True)
    save_name = _create_save_name(pc)

    npz_names = []
    for npz_name in os.listdir(save_dir):
        # TODO
        if npz_name.startswith(save_name) and '__y_' not in npz_name and '__base_' not in npz_name:
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

    renders_dir = os.path.join(pc.root_dir, 'renders')
    render_names = get_render_names(renders_dir,
                                    assert_unique=True,
                                    use_hashes=pc.use_hashes)
    log.info(f'{len(render_names)} renders found in {renders_dir}')

    log.info('Shuffling render names.')
    render_names = list(render_names)
    np.random.shuffle(render_names)

    mapping = {}
    if pc.use_hashes:
        mapping_path = os.path.join(renders_dir, 'mapping.txt')
        mapping = get_mapping(mapping_path)

    new_proc_render_names = []
    new_mels = []
    for render_name in tqdm(render_names):
        if render_name in proc_hashes:
            log.debug(f'{render_name} has already been processed.')
            continue

        if pc.use_hashes:
            audio_path = os.path.join(renders_dir, mapping[render_name])
        else:
            audio_path = os.path.join(renders_dir, render_name)
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
    new_mels = np.expand_dims(np.array(new_mels, dtype=np.float32), axis=-1)

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
             mels=mels,
             process_config=dict(pc._asdict()))

    if existing_npz_name:
        log.info(f'Deleting prev npz file: {existing_npz_name}')
        os.remove(os.path.join(save_dir, existing_npz_name))


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
            effect = param_to_effect[param]
            if effect.name in base_effect_names and desc not in exclude_descs:
                hash_tokens.append(param_str)

    render_hash = '__'.join(hash_tokens)

    if not render_hash:
        render_hash = 'dry'

    render_hash = f'{render_hash}.wav'
    return render_hash


def process_base_audio(orig_pc: ProcessConfig,
                       exclude_effects: Set[str] = None,
                       exclude_params: Set[int] = None) -> np.ndarray:
    if exclude_params is None:
        exclude_params = set()
    if exclude_effects is None:
        exclude_effects = set()

    log.info(f'Original root dir = {orig_pc.root_dir}')
    orig_save_dir = _create_save_dir(orig_pc, create_dirs=False)
    orig_save_name = _create_save_name(orig_pc)

    npz_names = []
    for npz_name in os.listdir(orig_save_dir):
        # TODO
        if npz_name.startswith(orig_save_name) and '__y_' not in npz_name and '__base_' not in npz_name:
            npz_names.append(npz_name)

    assert len(npz_names) == 1
    orig_proc_data = np.load(os.path.join(orig_save_dir, npz_names[0]))
    orig_proc_render_names = orig_proc_data['render_names'].tolist()
    log.info(f'Found {len(orig_proc_render_names)} original processed renders '
             f'in {npz_names[0]}')

    exclude_descs = generate_exclude_descs(exclude_effects, exclude_params)
    log.info(f'Exclude effects = {exclude_effects}')
    log.info(f'Exclude params = {exclude_params}')
    log.info(f'Exclude descs = {exclude_descs}')

    orig_root_dirs = os.path.normpath(orig_pc.root_dir).split(os.path.sep)
    orig_effect_dir_name = orig_root_dirs[-1]
    log.info(f'Original effect dir name: {orig_effect_dir_name}')

    orig_effect_dir_info = parse_save_name(orig_effect_dir_name, is_dir=True)
    gran = orig_effect_dir_info['gran']
    orig_effect_names = orig_effect_dir_info['name'].split('_')  # TODO
    log.info(f'Using granularity of {gran}')
    log.info(f'{orig_effect_names} original effects found.')

    base_effect_names = []
    for effect_name in orig_effect_names:
        effect = get_effect(effect_name)
        if not all(PARAM_TO_DESC[p] in exclude_descs for p in effect.order):
            base_effect_names.append(effect_name)
    if not base_effect_names:
        base_effect_names.append('dry')

    base_effect_names = sorted(list(set(base_effect_names)))
    log.info(f'{base_effect_names} base effects found.')

    # TODO
    if len(base_effect_names) > 2:
        base_use_hashes = True
    else:
        base_use_hashes = False

    base_effect_dir_name = f'{"_".join(base_effect_names)}__gran_{gran}'
    log.info(f'Base effect dir name: {base_effect_dir_name}')

    base_root_dir = os.path.normpath(os.path.join(orig_pc.root_dir, '../', base_effect_dir_name))
    log.info(f'Base root dir = {base_root_dir}')
    base_pc = ProcessConfig(hop_length=orig_pc.hop_length,
                            max_n_of_frames=orig_pc.max_n_of_frames,
                            n_fft=orig_pc.n_fft,
                            n_mels=orig_pc.n_mels,
                            normalize_audio=orig_pc.normalize_audio,
                            normalize_mel=orig_pc.normalize_mel,
                            sr=orig_pc.sr,
                            root_dir=base_root_dir,
                            use_hashes=base_use_hashes)
    base_save_dir = _create_save_dir(base_pc, create_dirs=False)
    base_save_name = _create_save_name(base_pc)

    npz_names = []
    for npz_name in os.listdir(base_save_dir):
        # TODO
        if npz_name.startswith(base_save_name) and '__y_' not in npz_name and '__base_' not in npz_name:
            npz_names.append(npz_name)

    assert len(npz_names) == 1
    base_proc_data = np.load(os.path.join(base_save_dir, npz_names[0]))
    base_proc_render_names = base_proc_data['render_names'].tolist()
    log.info(f'Found {len(base_proc_render_names)} base processed renders '
             f'in {npz_names[0]}')

    log.info('Loading reference base mels.')
    mels = base_proc_data['mels']

    render_name_to_mel = {}
    assert len(base_proc_render_names) == len(mels)
    for render_name, mel in zip(base_proc_render_names, mels):
        render_name_to_mel[render_name] = mel

    output_render_names = []
    output_mels = []
    for orig_render_name in tqdm(orig_proc_render_names):
        base_render_name = generate_base_render_hash(orig_render_name,
                                                     base_effect_names,
                                                     exclude_descs)
        output_render_names.append(base_render_name)
        output_mels.append(render_name_to_mel[base_render_name])

    log.info('Converting mels to ndarray.')
    output_mels = np.array(output_mels, dtype=np.float32)
    assert output_mels.shape == orig_proc_data['mels'].shape
    log.info(f'Output mels shape = {output_mels.shape}')

    # TODO
    # output_npz_name = f'{orig_save_name}__n_{len(output_render_names)}' \
    #                   f'___base__{"_".join(base_effect_names)}.npz'
    # log.info(f'Saving new npz file: {output_npz_name}')
    # np.savez(os.path.join(orig_save_dir, output_npz_name),
    #          render_names=output_render_names,
    #          mels=output_mels)
    return output_mels


def combine_mels(mels_paths: List[str],
                 save_path: str,
                 exclude_effects: Set[str] = None,
                 exclude_params: Set[int] = None) -> None:
    all_render_names = []
    all_mels = []
    all_base_mels = []
    for mels_path in mels_paths:
        data_npz_name = ntpath.basename(mels_path)
        data = np.load(mels_path, allow_pickle=True)
        render_names = data['render_names'].tolist()
        log.info(f'{len(render_names)} renders found in {data_npz_name}')

        mels = data['mels']
        log.info(f'mels shape = {mels.shape}')
        all_mels.append(mels)

        root_dir = os.path.normpath(os.path.join(os.path.split(mels_path)[0],
                                                 '../'))
        log.info(f'Extracted root_dir = {root_dir}')
        pc = data['process_config'].item()
        pc['root_dir'] = root_dir
        pc = ProcessConfig(**pc)
        base_mels = process_base_audio(pc,
                                       exclude_effects=exclude_effects,
                                       exclude_params=exclude_params)
        log.info(f'base_mels shape = {base_mels.shape}')
        all_base_mels.append(base_mels)

        all_render_names.extend(list(render_names))
        log.info(f'all_render_names length = {len(all_render_names)}')

    log.info(f'Concatenating mels.')
    all_mels = np.concatenate(all_mels, axis=0)
    log.info(f'Concatenating base mels.')
    all_base_mels = np.concatenate(all_base_mels, axis=0)
    log.info(f'all_mels shape = {all_mels.shape}')
    log.info(f'all_base_mels shape = {all_base_mels.shape}')
    log.info(f'Converting all_render_names to ndarray.')
    all_render_names = np.array(all_render_names)
    log.info(f'all_render_names shape = {all_render_names.shape}')

    assert len(all_mels) == len(all_base_mels) == len(all_render_names)

    rand_state = np.random.get_state()
    log.info('Shuffling all_mels.')
    np.random.shuffle(all_mels)

    np.random.set_state(rand_state)
    log.info('Shuffling all_base_mels.')
    np.random.shuffle(all_base_mels)

    np.random.set_state(rand_state)
    log.info('Shuffling all_render_names.')
    np.random.shuffle(all_render_names)

    log.info(f'all_mels shape = {all_mels.shape}')
    log.info(f'all_base_mels shape = {all_base_mels.shape}')
    log.info(f'all_render_names length = {len(all_render_names)}')

    log.info(f'Saving to {save_path}')
    np.savez(save_path,
             render_names=all_render_names,
             mels=all_mels,
             base_mels=all_base_mels)


if __name__ == '__main__':
    process_audio(os.path.join(CONFIGS_DIR, 'audio_process_test.yaml'))
    # process_base_audio(os.path.join(CONFIGS_DIR, 'audio_process_test.yaml'),
                       # exclude_effects={'distortion'})
                       # exclude_effects={'phaser'})
                       # exclude_effects={'flanger'})
                       # exclude_effects=set())
                       # exclude_effects={'distortion', 'phaser'})
                       # exclude_params={123})
    exit()

    # base_mels_path = '/Volumes/samsung_t5/reverse_synthesis'
    # base_mels_path = DATA_DIR
    # base_mels_path = os.path.join(base_mels_path, 'training/saw__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048')
    # mels_paths = [
    #     os.path.join(base_mels_path, 'compressor_distortion_eq_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'compressor_distortion_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'compressor_eq_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'compressor_distortion__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_49999.npz'),
    # ]
    # mels_paths = [
    #     os.path.join(base_mels_path, 'compressor_distortion_eq_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'distortion_eq_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'compressor_eq_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'eq_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    # ]
    # mels_paths = [
    #     os.path.join(base_mels_path, 'compressor_distortion_eq_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'distortion_eq_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'distortion_flanger_phaser__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_50000.npz'),
    #     os.path.join(base_mels_path, 'compressor_distortion__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_49999.npz'),
    # ]
    # combine_mels(mels_paths,
    #              os.path.join(DATA_DIR, 'combined_distortion_200k.npz'),
    #              exclude_effects={'distortion'})
    # exit()
