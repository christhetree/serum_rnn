import logging
import os
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from python.config import DEFAULT_DISTORTION, NOTE_LENGTH, \
    RENDER_LENGTH
from python.serum_util import setup_serum, set_preset

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


def create_distortion_data(
        preset_path: str,
        n_datapoints: int = 10000,
        note_length: float = NOTE_LENGTH,
        render_length: float = RENDER_LENGTH
) -> (np.ndarray, np.ndarray):
    n_modes = 14
    mode_values = list(range(n_modes))
    mode_values = [int((v / (n_modes - 1)) * 100) for v in mode_values]
    drive_values = list(range(0, 101))
    mix_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    vel_values = [127]
    midi_values = list(range(24, 85))

    n_combinations = len(drive_values) * len(mode_values) * len(mix_values) \
                     * len(vel_values) * len(midi_values)
    log.info(f'n_combinations = {n_combinations}')
    assert n_combinations > 10 * n_datapoints

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
    # preset_name = ntpath.basename(preset_path)
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
        # lr.output.write_wav(f'../out/{preset_name}_{str(param)}.wav', audio,
        #                     sr=RM_SR, norm=False)

    log.info(f'mode_dist = {mode_dist}')
    renders = np.array(renders, dtype=np.float64)
    params = np.array(params, dtype=np.int32)

    return renders, params
