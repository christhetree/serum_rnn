import hashlib
import logging
import os
from itertools import combinations
from typing import Set

import numpy as np
import yaml
from tqdm import tqdm

from audio_processing_util import ProcessConfig, create_save_name, \
    generate_base_render_hash, get_base_effect_info
from config import CONFIGS_DIR, DATASETS_DIR
from training_rnn import EFFECT_TO_IDX_MAPPING
from util import generate_exclude_descs, \
    parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def combine_mels_individual(mels_dir: str,
                            save_dir: str,
                            proc_dir_name: str,
                            exclude_effects: Set[str] = None,
                            exclude_all: bool = True,
                            include_base_effects: bool = False) -> None:
    _, mels_dir_name = os.path.split(mels_dir)

    existing_npz = set()
    for npz_name in os.listdir(mels_dir):
        if npz_name.endswith('.npz'):
            existing_npz.add(npz_name)

    log.info(f'{len(existing_npz)} npz found in {mels_dir}')

    if not os.path.exists(save_dir):
        log.info(f'Making new save dir for {save_dir}')
        os.makedirs(save_dir)
    else:
        log.info(f'Save dir {save_dir} already exists.')

    save_dir = os.path.join(save_dir, 'x')
    if not os.path.exists(save_dir):
        log.info(f'Making new save dir for {save_dir}')
        os.makedirs(save_dir)
    else:
        log.info(f'Save dir {save_dir} already exists.')

    effect_dir = os.path.normpath(
        os.path.join(os.path.split(mels_dir)[0], '../'))
    log.info(f'Extracted effect dir = {effect_dir}')

    preset_dir, effect_dir_name = os.path.split(effect_dir)

    _, preset_dir_name = os.path.split(preset_dir)
    log.info(f'Extracted preset dir name = {preset_dir_name}')

    preset_info = parse_save_name(preset_dir_name, is_dir=True)
    preset = preset_info['name']
    log.info(f'Extracted preset = {preset}')

    exclude_descs = generate_exclude_descs(exclude_effects)
    _, base_effect_names = get_base_effect_info(effect_dir_name,
                                                exclude_descs)
    if 'dry' in base_effect_names:
        base_effect_names.remove('dry')

    if exclude_all:
        exclude_descs_combos = []
        for n_effects in range(len(base_effect_names) + 1):
            for effects_combo in combinations(base_effect_names, n_effects):
                effects_combo = set(effects_combo)
                effects_combo.update(exclude_effects)
                exclude_descs_combos.append(
                    generate_exclude_descs(effects_combo))
    else:
        exclude_descs_combos = [exclude_descs]
    log.info(f'exclude_descs_combos len = {len(exclude_descs_combos)}')

    for npz_name in tqdm(existing_npz):
        npz_path = os.path.join(mels_dir, npz_name)
        render_name = np.load(npz_path)['render_name'].item()

        for exclude_descs_combo in exclude_descs_combos:
            base_effect_dir_name, base_effect_names = get_base_effect_info(
                effect_dir_name, exclude_descs_combo)

            base_render_name = generate_base_render_hash(render_name,
                                                         base_effect_names,
                                                         exclude_descs_combo)

            base_render_hash = hashlib.sha1(
                base_render_name.encode('utf-8')).hexdigest()
            base_npz_name = f'{base_render_hash}.npz'

            base_npz_path = os.path.join(preset_dir,
                                         base_effect_dir_name,
                                         proc_dir_name,
                                         mels_dir_name,
                                         base_npz_name)
            assert os.path.exists(base_npz_path)
            save_name = f'{preset}__{base_render_hash}__{npz_name}'
            save_path = os.path.join(save_dir, save_name)

            if include_base_effects:
                n_effects = len(EFFECT_TO_IDX_MAPPING)
                base_effects_tensor = np.zeros((n_effects,), dtype=np.float32)
                for base_effect_name in base_effect_names:
                    if base_effect_name == 'dry':
                        continue
                    effect_idx = EFFECT_TO_IDX_MAPPING[base_effect_name]
                    base_effects_tensor[effect_idx] = 1.0

                np.savez(save_path,
                         render_name=render_name,
                         mel_path=npz_path,
                         base_mel_path=base_npz_path,
                         base_effects=base_effects_tensor)
            else:
                np.savez(save_path,
                         render_name=render_name,
                         mel_path=npz_path,
                         base_mel_path=base_npz_path)


def combine_mels_all_combos(preset_dir: str,
                            pc: ProcessConfig,
                            save_dir: str,
                            exclude_effects: Set[str],
                            base_effects: Set[str] = None,
                            gran: int = 100) -> None:
    if base_effects is None:
        base_effects = set()
    assert exclude_effects
    assert all(e not in base_effects for e in exclude_effects)

    all_combos = []
    for n_effects in range(len(base_effects) + 1):
        for combo in combinations(base_effects, n_effects):
            combo = set(list(combo))
            combo.update(exclude_effects)
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
        processing_dir = os.path.join(effect_dir, pc.proc_dir_name)
        assert os.path.exists(processing_dir)
        mel_dir = os.path.join(processing_dir, mel_dir_name)
        assert os.path.exists(mel_dir)
        mels_dirs.append(mel_dir)

    log.info('mels_dirs:')
    for mels_dir in mels_dirs:
        log.info(f'-- {mels_dir}')
    log.info(f'Length of mels_dirs = {len(mels_dirs)}')

    for mels_dir in tqdm(mels_dirs):
        combine_mels_individual(mels_dir,
                                save_dir,
                                pc.proc_dir_name,
                                exclude_effects=exclude_effects)


if __name__ == '__main__':
    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)
    all_effects = {'compressor', 'distortion', 'eq', 'phaser', 'reverb-hall'}

    effect = 'compressor'
    # effect = 'distortion'
    # effect = 'eq'
    # effect = 'phaser'
    # effect = 'reverb-hall'

    exclude_effects = {effect}

    presets_cat = 'basic_shapes'
    presets = ['sine', 'triangle', 'saw', 'square']
    # presets_cat = 'adv_shapes'
    # presets = ['ld_power_5ths_[fp]', 'sy_mtron_saw_[sd]', 'sy_shot_dirt_stab_[im]', 'sy_vintage_bells_[fp]']
    # presets_cat = 'temporal'
    # presets = ['ld_iheardulike5ths_[sd]', 'ld_postmodern_talking_[fp]', 'sq_busy_lines_[lcv]', 'sy_runtheharm_[gs]']

    # renders_dir = DATA_DIR
    renders_dir = '/home/testacc/samsung_t5_local/reverse_synthesis'

    datasets_dir = DATASETS_DIR

    renders_dir = os.path.join(renders_dir, 'training_seq_5_v3')

    # save_name = f'testing__{effect}'
    # save_name = f'seq_5_v3__{presets_cat}__{effect}'
    save_name = f'seq_5_v3__proc__{presets_cat}__{effect}'

    save_dir = os.path.join(datasets_dir, save_name)
    base_effects = all_effects - exclude_effects

    preset_dirs = []
    for preset in presets:
        preset_dirs.append(os.path.join(renders_dir,
                                        f'{preset}__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048'))

    for preset_dir in preset_dirs:
        combine_mels_all_combos(preset_dir,
                                pc,
                                save_dir,
                                exclude_effects,
                                base_effects,
                                gran=100)  # TODO
