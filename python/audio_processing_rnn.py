import hashlib
import logging
import os
from itertools import combinations
from typing import List, Set

import numpy as np
import yaml
from tqdm import tqdm

from audio_processing_util import ProcessConfig, get_base_effect_info, \
    generate_base_render_hash, create_save_name
from config import DATASETS_DIR, CONFIGS_DIR, DATA_DIR
from util import parse_save_name, generate_exclude_descs

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def create_sequence(orig_render_name: str,
                    effect_names: List[str],
                    preset_dir: str,
                    orig_effect_dir_name: str,
                    mels_dir_name: str,
                    orig_npz_name: str) -> (List[str], List[str], List[str]):
    assert effect_names != ['dry']
    render_name_seq = [orig_render_name]
    orig_mel_path = os.path.join(preset_dir,
                                 orig_effect_dir_name,
                                 'processing',  # TODO
                                 mels_dir_name,
                                 orig_npz_name)
    assert os.path.exists(orig_mel_path)
    mel_path_seq = [orig_mel_path]
    effect_seq = []

    prev_effect_dir_name = orig_effect_dir_name
    prev_render_name = orig_render_name
    for effect_name in effect_names:
        effect_seq.append(effect_name)
        exclude_descs = generate_exclude_descs({effect_name})

        base_effect_dir_name, base_effect_names = get_base_effect_info(
            prev_effect_dir_name, exclude_descs)
        base_render_name = generate_base_render_hash(prev_render_name,
                                                     base_effect_names,
                                                     exclude_descs)
        base_render_hash = hashlib.sha1(
            base_render_name.encode('utf-8')).hexdigest()
        base_npz_name = f'{base_render_hash}.npz'

        base_mel_path = os.path.join(preset_dir,
                                     base_effect_dir_name,
                                     'processing',  # TODO
                                     mels_dir_name,
                                     base_npz_name)
        assert os.path.exists(base_mel_path)
        mel_path_seq.append(base_mel_path)
        render_name_seq.append(base_render_name)

        prev_effect_dir_name = base_effect_dir_name
        prev_render_name = base_render_name

    effect_seq.append('dry')
    effect_seq.reverse()
    mel_path_seq.reverse()
    render_name_seq.reverse()
    assert len(effect_seq) == len(mel_path_seq) == len(render_name_seq)

    return effect_seq, mel_path_seq, render_name_seq


def create_effect_sequences(mels_dir: str,
                            save_dir: str) -> None:
    _, mels_dir_name = os.path.split(mels_dir)

    existing_npz = set()
    for npz_name in os.listdir(mels_dir):
        if npz_name.endswith('.npz'):
            existing_npz.add(npz_name)

    log.info(f'{len(existing_npz)} npz found in {mels_dir}')
    if len(existing_npz) == 0:
        return

    if not os.path.exists(save_dir):
        log.info(f'Making new save dir for {save_dir}')
        os.makedirs(save_dir)
    else:
        log.info(f'Save dir {save_dir} already exists.')

    effect_dir = os.path.normpath(
        os.path.join(os.path.split(mels_dir)[0], '../'))
    log.info(f'Extracted effect dir = {effect_dir}')

    preset_dir, effect_dir_name = os.path.split(effect_dir)
    calc_effect_dir_name, effect_names = get_base_effect_info(effect_dir_name)
    assert calc_effect_dir_name == effect_dir_name
    log.info(f'Extracted effect names = {effect_names}')

    _, preset_dir_name = os.path.split(preset_dir)
    log.info(f'Extracted preset dir name = {preset_dir_name}')

    preset_info = parse_save_name(preset_dir_name, is_dir=True)
    preset = preset_info['name']
    log.info(f'Extracted preset = {preset}')

    for npz_name in tqdm(existing_npz):
        npz_path = os.path.join(mels_dir, npz_name)
        render_name = np.load(npz_path)['render_name'].item()
        np.random.shuffle(effect_names)
        effect_seq, mel_path_seq, render_name_seq = create_sequence(
            render_name,
            effect_names,
            preset_dir,
            effect_dir_name,
            mels_dir_name,
            npz_name
        )
        all_render_names = ''.join(render_name_seq)
        render_name_seq_hash = hashlib.sha1(
            all_render_names.encode('utf-8')).hexdigest()
        save_name = f'{preset}__seq_len_{len(mel_path_seq)}' \
                    f'__{render_name_seq_hash}.npz'
        save_path = os.path.join(save_dir, save_name)
        np.savez(save_path,
                 effect_seq=effect_seq,
                 mel_path_seq=mel_path_seq)


def create_effect_sequences_all_combos(preset_dir: str,
                                       pc: ProcessConfig,
                                       save_dir: str,
                                       all_effects: Set[str] = None,
                                       gran: int = 100) -> None:
    all_combos = []
    for n_effects in range(1, len(all_effects) + 1):
        for combo in combinations(all_effects, n_effects):
            combo = set(list(combo))
            all_combos.append(combo)

    log.info(f'All effect combos = {all_combos}')
    log.info(f'Len of effect combos = {len(all_combos)}')

    mel_dir_name = create_save_name(pc)

    mels_dirs = []
    for combo in tqdm(all_combos):
        effects = sorted(list(combo))
        effect_dir = os.path.join(preset_dir,
                                  f'{"_".join(effects)}__gran_{gran}')
        assert os.path.exists(effect_dir)
        processing_dir = os.path.join(effect_dir, 'processing')
        assert os.path.exists(processing_dir)
        mel_dir = os.path.join(processing_dir, mel_dir_name)
        assert os.path.exists(mel_dir)
        mels_dirs.append(mel_dir)

    log.info('mels_dirs:')
    for mels_dir in mels_dirs:
        log.info(f'-- {mels_dir}')
    log.info(f'Length of mels_dirs = {len(mels_dirs)}')

    for mels_dir in tqdm(mels_dirs):
        create_effect_sequences(mels_dir, save_dir)


if __name__ == '__main__':
    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)

    all_effects = {'compressor', 'distortion', 'eq', 'phaser', 'reverb-hall'}

    presets_cat = 'basic_shapes'
    presets = ['sine', 'triangle', 'saw', 'square']
    # presets_cat = 'adv_shapes'
    # presets = ['ld_power_5ths_[fp]', 'sy_mtron_saw_[sd]', 'sy_shot_dirt_stab_[im]', 'sy_vintage_bells_[fp]']
    # presets_cat = 'temporal'
    # presets = ['ld_iheardulike5ths_[sd]', 'ld_postmodern_talking_[fp]', 'sq_busy_lines_[lcv]', 'sy_runtheharm_[gs]']

    # renders_dir = DATA_DIR
    renders_dir = '/home/testacc/samsung_t5_local/reverse_synthesis'

    datasets_dir = DATASETS_DIR

    # renders_dir = os.path.join(renders_dir, 'training_seq_5_v3_local')
    renders_dir = os.path.join(renders_dir, 'training_seq_5_v3')

    # save_name = f'seq_5_v3_local__proc__{presets_cat}__rnn'
    save_name = f'seq_5_v3__proc__{presets_cat}__rnn'

    save_dir = os.path.join(datasets_dir, save_name)

    preset_dirs = []
    for preset in presets:
        preset_dirs.append(os.path.join(renders_dir,
                                        f'{preset}__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048'))

    for preset_dir in preset_dirs:
        create_effect_sequences_all_combos(preset_dir,
                                           pc,
                                           save_dir,
                                           all_effects,
                                           gran=100)
