import logging
import os
from typing import Any, Dict

import gym
import numpy as np
import soundfile as sf
from gym.spaces import Discrete, Box
from sklearn.metrics import mean_squared_error
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from python.audio_processing import get_mel_spec
from python.config import DEFAULT_SERUM_PRESET_PATH, DEFAULT_DISTORTION, RM_SR
from python.serum_util import setup_serum, set_preset

# logging.basicConfig(level=os.environ.get('LOGLEVEL', 'DEBUG'))
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class SerumEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    N_DISCRETE_ACTIONS = 3
    UP = 0
    DOWN = 1
    NO_OP = 2
    OBV_LENGTH = 44544
    STEP = 0.01

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 param_idx: int = 97,  # Drive
                 param_gold_v: float = 0.5,
                 midi_v: int = 40,
                 vel_v: int = 127,
                 note_length: float = 1.0,
                 render_length: float = 1.0) -> None:
        super(SerumEnv, self).__init__()
        self.action_space = Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = Box(low=-1.0,
                                     high=1.0,
                                     shape=(self.OBV_LENGTH,))
        self.engine = setup_serum(DEFAULT_SERUM_PRESET_PATH, render_once=True)
        set_preset(self.engine, DEFAULT_DISTORTION)

        self.param_idx = param_idx
        self.param_gold_v = param_gold_v
        self.param_curr_v = np.random.randint(0, 101) / 100
        self.midi_v = midi_v
        self.vel_v = vel_v
        self.note_length = note_length
        self.render_length = render_length

        self.engine.set_parameter(self.param_idx, self.param_gold_v)
        self.gold_audio = self._render_audio(self.midi_v, self.vel_v)
        self.gold_mel_spec = get_mel_spec(self.gold_audio,
                                          normalize_audio=False,
                                          normalize_mel=True)

        # sf.write(f'../out/gold.wav', self.gold_audio, RM_SR)
        # for idx in range(11):
        #     self.engine.set_parameter(self.param_idx, idx / 10)
        #     audio = self._render_audio(self.midi_v, self.vel_v)
        #     sf.write(f'../out/{idx}.wav', audio, RM_SR)
        #     mel_spec = get_mel_spec(audio,
        #                             normalize_audio=False,
        #                             normalize_mel=True)
        #     mse = mean_squared_error(mel_spec, self.gold_mel_spec)
        #     print(f'{idx:2}: {mse:.6f}')

        self.curr_audio = None
        self.curr_mel_spec = None

    def _render_audio(self, midi_v, vel_v) -> np.ndarray:
        self.engine.render_patch(midi_v,
                                 vel_v,
                                 self.note_length,
                                 self.render_length,
                                 False)
        audio = np.array(self.engine.get_audio_frames(), dtype=np.float32)
        return audio

    def step(self, action: int) -> (np.ndarray, float, bool, Dict[Any, Any]):
        log.info('step')
        if action == self.UP:
            log.info('Action = up')
            self.param_curr_v += self.STEP
            self.param_curr_v = min(self.param_curr_v, 1.0)
        elif action == self.DOWN:
            log.info('Action = down')
            self.param_curr_v -= self.STEP
            self.param_curr_v = max(self.param_curr_v, 0.0)
        elif action == self.NO_OP:
            log.info('Action = no-op')
        else:
            raise ValueError(f'Invalid action: {action}')

        log.info(f'param_curr_v = {self.param_curr_v}')
        self.engine.set_parameter(self.param_idx, self.param_curr_v)
        self.curr_audio = self._render_audio(self.midi_v, self.vel_v)
        self.curr_mel_spec = get_mel_spec(self.curr_audio,
                                          normalize_audio=False,
                                          normalize_mel=True)
        mse = mean_squared_error(self.curr_mel_spec, self.gold_mel_spec)
        log.info(f'mse = {mse:.8f}')

        done = False
        if mse < 0.001:
            log.info('done')
            done = True

        # reward = -float(mse)
        # reward = 0.5 - abs(0.5 - self.param_curr_v)
        if done:
            reward = 1.0
        else:
            reward = 0.0

        log.info(f'reward = {reward}')
        return self.curr_audio, reward, done, {}

    def reset(self) -> np.ndarray:
        log.info('reset')
        set_preset(self.engine, DEFAULT_DISTORTION)
        self.param_curr_v = np.random.randint(0, 101) / 100
        self.engine.set_parameter(self.param_idx, self.param_curr_v)
        audio = self._render_audio(self.midi_v, self.midi_v)
        return audio

    def render(self, mode: str = 'human') -> None:
        log.info('rendering')

    def close(self) -> None:
        pass


def example() -> None:
    # env = gym.make('CartPole-v1')
    env = SerumEnv()
    # check_env(env, warn=True)
    exit()

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    example()
