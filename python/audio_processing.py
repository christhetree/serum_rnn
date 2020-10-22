import hashlib
import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Set

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from audio_features import get_mel_spec
from audio_processing_util import ProcessConfig, create_save_dir
from config import CONFIGS_DIR
from util import get_render_names, get_mapping

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def process_audio_individual(pc: ProcessConfig) -> None:
    save_dir = create_save_dir(pc, create_dirs=True)

    existing_npz = set()
    for npz_name in os.listdir(save_dir):
        existing_npz.add(npz_name)

    log.info(f'Found {len(existing_npz)} existing processed renders.')

    renders_dir = os.path.join(pc.root_dir, 'renders')
    render_names = get_render_names(renders_dir,
                                    assert_unique=True,
                                    use_hashes=pc.use_hashes)
    log.info(f'{len(render_names)} renders found in {renders_dir}')

    mapping = {}
    if pc.use_hashes:
        mapping_path = os.path.join(renders_dir, 'mapping.txt')
        mapping = get_mapping(mapping_path)

    feature_keys = []
    means = defaultdict(list)
    stds = defaultdict(list)
    maxs = defaultdict(list)
    mins = defaultdict(list)

    for render_name in tqdm(render_names):
        render_hash = hashlib.sha1(render_name.encode('utf-8')).hexdigest()
        npz_name = f'{render_hash}.npz'

        if npz_name in existing_npz:
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

        audio_features = get_mel_spec(audio,
                                      sr=pc.sr,
                                      n_fft=pc.n_fft,
                                      hop_length=pc.hop_length,
                                      n_mels=pc.n_mels,
                                      max_n_of_frames=pc.max_n_of_frames,
                                      norm_audio=pc.norm_audio,
                                      norm_mel=pc.norm_mel,
                                      fmin=pc.fmin,
                                      fmax=pc.fmax,
                                      db_ref=pc.db_ref,
                                      top_db=pc.top_db,
                                      n_mfcc=pc.n_mfcc,
                                      calc_cent=pc.calc_cent,
                                      calc_bw=pc.calc_bw,
                                      calc_flat=pc.calc_flat)

        audio_features = dict(audio_features._asdict())
        feature_keys = audio_features.keys()
        for feature, value in audio_features.items():
            means[feature].append(np.mean(value))
            stds[feature].append(np.std(value))
            maxs[feature].append(np.max(value))
            mins[feature].append(np.min(value))

        save_path = os.path.join(save_dir, npz_name)
        np.savez(save_path, render_name=render_name, **audio_features)
        existing_npz.add(npz_name)

    for feature in feature_keys:
        log.info(f'{feature}: processed {len(means[feature])} new renders.')
        if len(means[feature]):
            log.info(f'{feature} mean = {np.mean(means[feature]):.5f}')
            log.info(f'{feature} mean of stds = {np.mean(stds[feature]):.5f}')
            log.info(f'{feature} max = {np.max(means[feature]):.5f}')
            log.info(f'{feature} min = {np.min(means[feature]):.5f}')


def process_audio_all_combos(orig_pc: ProcessConfig,
                             effects: Set[str],
                             gran: int = 100) -> None:
    all_combos = []
    for n_effects in range(1, len(effects) + 1):
        for combo in combinations(effects, n_effects):
            all_combos.append(set(list(combo)))
    all_combos.append({'dry'})
    all_combos.reverse()

    log.info(f'All effect combos = {all_combos}')
    log.info(f'Len of effect combos = {len(all_combos)}')

    preset_dir = os.path.split(orig_pc.root_dir)[0]

    for combo in tqdm(all_combos):
        if len(combo) > 2:  # TODO
            use_hashes = True
        else:
            use_hashes = False

        effects = sorted(list(combo))
        root_dir = os.path.join(preset_dir, f'{"_".join(effects)}__gran_{gran}')
        pc = orig_pc._replace(use_hashes=use_hashes, root_dir=root_dir)
        process_audio_individual(pc)


if __name__ == '__main__':
    process_config_path = os.path.join(CONFIGS_DIR, 'audio_process_test.yaml')
    with open(process_config_path, 'r') as config_f:
        process_config = yaml.full_load(config_f)
    pc = ProcessConfig(**process_config)
    all_effects = {'compressor', 'distortion', 'eq', 'phaser', 'reverb-hall'}

    process_audio_all_combos(pc, all_effects)
