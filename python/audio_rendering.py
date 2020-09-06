import logging
import ntpath
import os
import random
from collections import defaultdict, namedtuple
from typing import List, Dict

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from python.config import DEFAULT_DISTORTION, NOTE_LENGTH, \
    RENDER_LENGTH, RM_SR, CONFIGS_DIR, \
    Effect, distortion, flanger, DESC_TO_PARAM, PARAM_TO_DESC
from python.serum_util import setup_serum, set_preset

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'DEBUG'))
log = logging.getLogger(__name__)

RenderConfig = namedtuple(
    'RenderConfig',
    'root_dir n preset note_length render_length midi vel granularity effects'
)


class PatchGenerator:
    def __init__(self, rc: RenderConfig) -> None:
        super().__init__()
        self.granularity = rc.granularity

        n_combos = 1
        params = set()
        effect_names = set()
        param_choices = {}
        param_n_choices = {}
        param_defaults = {}

        for effect_render_data in rc.effects:
            effect_name = effect_render_data['name']
            if effect_name in effect_names:
                log.warning(f'Duplicate effect "{effect_name}" in config.')
                continue

            effect_names.add(effect_name)
            effect = get_effect(effect_name)

            for desc, choices in effect_render_data.items():
                if desc == 'name':
                    continue

                if desc not in DESC_TO_PARAM:
                    raise KeyError(f'{desc} not a valid param description.')

                param = DESC_TO_PARAM[desc]
                params.add(param)
                default_value = effect.default[param]
                default = int((default_value * self.granularity) + 0.5)
                param_defaults[param] = default

                if effect.binary and param in effect.binary:
                    n_choices = 2
                elif effect.categorical and param in effect.categorical:
                    n_choices = effect.categorical[param]
                elif effect.continuous and param in effect.continuous:
                    n_choices = rc.granularity + 1
                else:
                    raise KeyError(
                        f'{desc} param could not be found in {effect.name}.')

                len_choices = len(choices)
                if len_choices > 0:
                    n_combos *= len_choices
                    param_choices[param] = choices
                else:
                    n_combos *= n_choices
                    param_choices[param] = n_choices

                param_n_choices[param] = n_choices

        curr_params = sorted(list(params))
        effect_names = sorted(list(effect_names))

        curr_patch = {}
        default_diff = {}
        for param, choices in param_choices.items():
            if type(choices) is list:
                choice = np.random.choice(choices)
            else:
                choice = np.random.randint(0, choices)

            if choice != param_defaults[param]:
                default_diff[param] = choice

            n_choices = param_n_choices[param]
            param_v = float(choice / (n_choices - 1))
            curr_patch[param] = param_v

        self.n_combos = n_combos
        self.params = params
        self.effect_names = effect_names
        self.curr_params = curr_params
        self.param_choices = param_choices
        self.param_n_choices = param_n_choices
        self.param_defaults = param_defaults
        self.default_diff = default_diff
        self.curr_patch = curr_patch

    def generate_random_patch(
            self,
            n_changes: int = 1
    ) -> (Dict[int, int], Dict[int, float]):
        for param in self.curr_params[:n_changes]:
            choices = self.param_choices[param]
            if type(choices) is list:
                choice = np.random.choice(choices)
            else:
                choice = np.random.randint(0, choices)

            if choice != self.param_defaults[param]:
                self.default_diff[param] = choice
            elif param in self.default_diff:
                del self.default_diff[param]

            n_choices = self.param_n_choices[param]
            param_v = float(choice / (n_choices - 1))
            self.curr_patch[param] = param_v

        if n_changes < len(self.curr_params):
            self.curr_params = self.curr_params[n_changes:] \
                               + self.curr_params[:n_changes]

        return self.default_diff, self.curr_patch


def get_effect(name: str) -> Effect:
    if name == 'distortion':
        return distortion
    elif name == 'flanger':
        return flanger
    else:
        raise ValueError


def generate_render_hash(effect_names: List[str],
                         default_diff: Dict[int, int],
                         param_n_digits: Dict[int, int]) -> str:
    hash_tokens = ['_'.join(effect_names)]

    for param, choice in sorted(default_diff.items()):
        desc = PARAM_TO_DESC[param]
        n_digits = param_n_digits[param]
        hash_tokens.append(f'{desc}_{choice:0{n_digits}}')

    render_hash = '__'.join(hash_tokens)
    render_hash = f'{render_hash}.wav'
    return render_hash


def render_audio(render_config_path: str) -> None:
    with open(render_config_path, 'r') as config_f:
        render_config = yaml.full_load(config_f)

    rc = RenderConfig(**render_config)

    if not os.path.exists(rc.root_dir):
        os.makedirs(rc.root_dir)
    else:
        log.info(f'Root dir {rc.root_dir} already exists.')

    preset_name = os.path.splitext(ntpath.basename(rc.preset))[0]
    int_dir_name = f'{preset_name}__nl_{rc.note_length:.2f}__rl_' \
                   f'{rc.render_length:.2f}__vel_{rc.vel:03}__midi_{rc.midi:03}'

    save_dir = os.path.join(rc.root_dir, int_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        log.info(f'Save dir {int_dir_name} already exists.')

    render_names = set()
    for render_name in os.listdir(save_dir):
        if render_name.endswith('.wav'):
            render_names.add(render_name)

    init_n_renders = len(render_names)
    log.info(f'{init_n_renders} existing renders found.')

    engine = setup_serum(rc.preset, render_once=True)
    patch_gen = PatchGenerator(rc)
    effect_names = patch_gen.effect_names
    param_n_digits = {k: len(str(v))
                      for k, v in patch_gen.param_n_choices.items()}

    log.info(f'{patch_gen.n_combos} no. of possible rendering combos.')
    log.info(f'{effect_names} effects in patch generator.')

    for effect_render_data in rc.effects:
        effect = get_effect(effect_render_data['name'])
        log.info(f'Setting default {effect.name} params.')
        set_preset(engine, effect.default)

    n_rendered = 0
    pbar = tqdm(total=rc.n)
    while n_rendered < rc.n:
        default_diff, patch = patch_gen.generate_random_patch()
        render_name = generate_render_hash(effect_names,
                                           default_diff,
                                           param_n_digits)

        if render_name in render_names:
            log.debug(f'Duplicate render generated: {render_name}')
        else:
            set_preset(engine, patch)
            engine.render_patch(rc.midi,
                                rc.vel,
                                rc.note_length,
                                rc.render_length,
                                False)
            audio = np.array(engine.get_audio_frames(), dtype=np.float64)
            save_path = os.path.join(save_dir, render_name)
            sf.write(save_path, audio, RM_SR)

            render_names.add(render_name)
            n_rendered += 1
            pbar.update(1)

    derp = 1


def create_distortion_data(
        preset_path: str,
        n_datapoints: int = 100,
        note_length: float = NOTE_LENGTH,
        render_length: float = RENDER_LENGTH
) -> (np.ndarray, np.ndarray):
    n_modes = 14
    mode_values = list(range(n_modes))
    mode_values = [int((v / (n_modes - 1)) * 100) for v in mode_values]
    drive_values = list(range(0, 101))
    # mix_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mix_values = [100]
    vel_values = [127]
    # midi_values = list(range(24, 85))
    midi_values = [40]

    n_combinations = len(drive_values) * len(mode_values) * len(mix_values) \
                     * len(vel_values) * len(midi_values)
    log.info(f'n_combinations = {n_combinations}')
    # assert n_combinations > 4 * n_datapoints

    params = set()
    while len(params) < n_datapoints:
        mode_v = random.choice(mode_values)
        drive_v = random.choice(drive_values)
        mix_v = random.choice(mix_values)
        vel_v = random.choice(vel_values)
        midi_v = random.choice(midi_values)
        param = (mode_v, drive_v, mix_v, vel_v, midi_v)
        params.add(param)

    params = list(params)
    preset_name = ntpath.basename(preset_path)
    mode_dist = defaultdict(int)
    renders = []

    engine = setup_serum(preset_path, render_once=True)
    set_preset(engine, DEFAULT_DISTORTION)

    for param in tqdm(params):
        mode_v, drive_v, mix_v, vel_v, midi_v = param
        mode_dist[mode_v] += 1

        engine.set_parameter(99, mode_v / 100)
        engine.set_parameter(97, drive_v / 100)
        engine.set_parameter(96, mix_v / 100)
        engine.render_patch(midi_v, vel_v, note_length, render_length, False)
        audio = np.array(engine.get_audio_frames(), dtype=np.float64)
        renders.append(audio)
        # sf.write(f'../out/{preset_name}_{str(param)}.wav', audio, RM_SR)

    log.info(f'mode_dist = {mode_dist}')
    renders = np.array(renders, dtype=np.float64)
    params = np.array(params, dtype=np.int32)

    return renders, params


if __name__ == '__main__':
    render_audio(os.path.join(CONFIGS_DIR, 'audio_render_test.yaml'))
    # renders, params = create_distortion_data(DEFAULT_SERUM_PRESET_PATH)
    # np.savez('../data/datasets/renders_10k_testing_midi.npz', renders=renders, params=params)
