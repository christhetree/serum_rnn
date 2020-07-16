import json
import logging
import ntpath
import os
from typing import Dict

import librenderman as rm
from tqdm import tqdm

from python.config import RM_SR, RM_BUFFER_SIZE, RM_FFT_SIZE, SERUM_PATH, \
    PRESETS_DIR

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


def load_preset(engine: rm.RenderEngine,
                preset_path: str,
                render_once: bool = True) -> None:
    engine.load_preset(preset_path)

    if render_once:
        engine.render_patch(60, 127, 1.0, 3.0, False)


def setup_serum(preset_path: str = None,
                sr: int = RM_SR,
                render_once: bool = True) -> rm.RenderEngine:
    engine = rm.RenderEngine(sr, RM_BUFFER_SIZE, RM_FFT_SIZE)
    engine.load_plugin(SERUM_PATH)

    if preset_path:
        load_preset(engine, preset_path, render_once=render_once)

    return engine


def find_fxp_differences(path_a: str, path_b: str) -> None:
    engine_a = setup_serum(path_a)
    engine_b = setup_serum(path_b)

    assert engine_a.get_plugin_parameters_description() \
           == engine_b.get_plugin_parameters_description()

    param_desc_str = engine_a.get_plugin_parameters_description()
    param_desc = {}
    for desc in param_desc_str.strip().split('\n'):
        k, v = desc.split(': ')
        k = int(k.strip())
        param_desc[k] = v.strip()

    differences = []

    for param_idx in range(engine_a.get_plugin_parameter_size()):
        val_a = engine_a.get_parameter(param_idx)
        val_b = engine_b.get_parameter(param_idx)

        if val_a != val_b:
            param_name = param_desc[param_idx]
            differences.append((param_idx, param_name, val_a, val_b))

    log.info(f'Found the following differences: {differences}')


def load_presets(path: str) -> Dict[str, Dict[int, float]]:
    with open(path, 'r') as input_f:
        presets_str = json.load(input_f)

    presets = {}
    for preset_name, preset in presets_str.items():
        presets[preset_name] = {int(k): v for k, v in preset.items()}

    return presets


def set_preset(engine: rm.RenderEngine, preset: Dict[int, float]) -> None:
    for param_idx, param_value in preset.items():
        engine.set_parameter(param_idx, param_value)


def crawl_presets(root_dir: str,
                  save_name: str,
                  save_dir: str = PRESETS_DIR,
                  file_ending: str = '.fxp') -> None:
    save_path = os.path.join(save_dir, save_name)

    preset_paths = []
    for root, dirs, files in tqdm(os.walk(root_dir)):
        for f in files:
            if f.endswith(file_ending):
                preset_paths.append(os.path.join(root, f))

    log.info(f'Found {len(preset_paths)} files ending with {file_ending}')
    presets = {}

    for preset_path in tqdm(preset_paths):
        preset_name = ntpath.basename(preset_path)
        engine = setup_serum(preset_path, render_once=False)
        n_params = engine.get_plugin_parameter_size()

        preset = {param_idx: engine.get_parameter(param_idx)
                  for param_idx in range(n_params)}

        if preset_name not in presets:
            presets[preset_name] = preset
        else:
            log.warning(f'Duplicate preset found: {preset_name}')

    log.info(f'Length of presets is {len(presets)}')

    with open(save_path, 'w') as output_f:
        json.dump(presets, output_f, indent=4, sort_keys=True)
