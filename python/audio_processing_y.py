import logging
import ntpath
import os
from collections import defaultdict
from typing import Set

import numpy as np
from tqdm import tqdm

from effects import param_to_type, DESC_TO_PARAM, \
    param_to_effect, PARAM_TO_DESC
from util import parse_save_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def generate_y(path: str,
               params: Set[int],
               gran: int = 100) -> None:
    assert os.path.isfile(path)

    data_npz_name = os.path.splitext(ntpath.basename(path))[0]
    # effect_dir_name = os.path.normpath(path).split(os.path.sep)[-3]
    # log.info(f'.npz file name: {data_npz_name}')
    # log.info(f'Effect dir name: {effect_dir_name}')
    #
    # effect_dir_info = parse_save_name(effect_dir_name, is_dir=True)
    # gran = effect_dir_info['gran']
    # effect_names = effect_dir_info['name'].split('_')  # TODO
    # log.info(f'Using granularity of {gran}')
    # log.info(f'{effect_names} effects found.')
    #
    # if params is None:
    #     log.info('No params provided. Calculating y for all params.')
    #     params = set()
    #     for effect_name in effect_names:
    #         effect = get_effect(effect_name)
    #         for param in effect.order:
    #             params.add(param)

    params = sorted(list(params))
    log.info(f'Calculating y for the following params: {params}')

    param_types = defaultdict(list)
    for param in params:
        param_types[param_to_type[param]].append(param)
    binary_params = sorted(param_types['binary'])
    categorical_params = sorted(param_types['categorical'])
    continuous_params = sorted(param_types['continuous'])

    log.info(f'Binary params: {binary_params}')
    log.info(f'Categorical params: {categorical_params}')
    log.info(f'Continuous params: {continuous_params}')

    data = np.load(path, allow_pickle=True)
    render_names = data['render_names'].tolist()
    log.info(f'{len(render_names)} renders found in {data_npz_name}')
    y = defaultdict(list)

    for render_name in tqdm(render_names):
        render_name_info = parse_save_name(render_name, is_dir=False)
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
                cont_values.append(render_name_params[param] / gran)
            else:
                effect = param_to_effect[param]
                cont_values.append(effect.default[param])

        if cont_values:
            y['continuous'].append(cont_values)

    y = dict(y)
    y['gran'] = gran

    log.info('Converting to ndarray.')
    for key in tqdm(y):
        if key == 'binary' or key == 'continuous':
            y[key] = np.array(y[key], dtype=np.float32)
        else:
            y[key] = np.array(y[key], dtype=np.int32)

        log.info(f'{key} ndarray shape: {y[key].shape}')

    n_categories = []
    param_to_desc = []
    for param in categorical_params:
        effect = param_to_effect[param]
        n_classes = effect.categorical[param]
        desc = PARAM_TO_DESC[param]
        param_to_desc.append(desc)
        n_categories.append(n_classes)

    log.info(f'n_categories = {n_categories}')
    if binary_params:
        y['binary_params'] = np.array(binary_params, dtype=np.int32)
    if categorical_params:
        y['categorical_params'] = np.array(categorical_params, dtype=np.int32)
        y['n_categories'] = np.array(n_categories, dtype=np.int32)
        y['param_to_desc'] = np.array(param_to_desc)
    if continuous_params:
        y['continuous_params'] = np.array(continuous_params, dtype=np.int32)

    save_dir = os.path.split(path)[0]
    save_name = f'{data_npz_name}__y_{"_".join(str(p) for p in params)}.npz'
    save_path = os.path.join(save_dir, save_name)
    log.info(f'Saving as {save_name}')
    np.savez(save_path, **y)


if __name__ == '__main__':
    # n = 56
    # n = 1000
    # n = 25000
    # gran = 1000
    gran = 100
    # effect = 'chorus'
    # params = {118, 119, 120, 121, 122, 123}
    # effect = 'compressor'
    # params = {270, 271, 272}
    # effect = 'distortion'
    params = {97, 99}
    # effect = 'eq'
    # params = {88, 89, 90, 91, 92, 93, 94, 95}
    # effect = 'filter'
    # params = {142, 143, 144, 145, 146, 268}
    # effect = 'flanger'
    # params = {105, 106, 107, 108}
    # effect = 'phaser'
    # params = {111, 112, 113, 114, 115}
    # effect = 'reverb-hall'
    # params = {82, 83, 84, 85, 86, 87}
    # effect = 'distortion_phaser'

    # params = {88, 89, 90, 91}

    # generate_y(os.path.join(DATA_DIR, f'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}.npz'),
    # generate_y(os.path.join(DATA_DIR, f'audio_test_data/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}.npz'),
    # generate_y(os.path.join(DATA_DIR, 'combined_compressor_200k.npz'),
    # generate_y(os.path.join(DATA_DIR, 'combined_eq_200k.npz'),
    # generate_y(os.path.join(DATA_DIR, 'combined_distortion_200k.npz'),
    #            gran,
    #            params)
