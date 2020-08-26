import logging
import os
from typing import Dict, Any

import gym
import numpy as np
from gym.spaces import Box
from stable_baselines3 import PPO, A2C
from stable_baselines3.common import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch as th

# logging.basicConfig(level=os.environ.get('LOGLEVEL', 'DEBUG'))
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


class TestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 target_v: float = 0.5,
                 n_params: int = 1,
                 max_ep_steps: int = 10) -> None:
        super(TestEnv, self).__init__()
        self.n_params = n_params
        self.max_ep_steps = max_ep_steps
        self.curr_step_idx = 0

        self.action_space = Box(low=-0.1,
                                high=0.1,
                                shape=(self.n_params,),
                                dtype=np.float32)
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(self.n_params,),
                                     dtype=np.float32)
        self.reward_range = (-1.0, 1.0)
        self.target_v = np.full((self.n_params,), target_v, dtype=np.float32)
        self.curr_v = np.random.uniform(0.0,
                                        1.0,
                                        (self.n_params,)).astype(np.float32)

    def step(self,
             action: np.ndarray) -> (np.ndarray, float, bool, Dict[Any, Any]):
        prev_v = self.curr_v
        self.curr_v = np.clip(self.curr_v + action, 0.0, 1.0)
        self.curr_step_idx += 1

        diff = self.target_v - self.curr_v
        mae = np.mean(np.abs(diff))
        # mse = np.mean(np.square(diff))
        reward = (0.5 - mae) * 2

        if self.curr_step_idx >= self.max_ep_steps:
            return self.curr_v, reward, True, {'stuck': True}

        if np.allclose(prev_v, self.curr_v):
            return self.curr_v, -1.0, True, {'stuck': True}

        done = False
        if mae < 0.05:
            done = True

        return self.curr_v, reward, done, {'stuck': False}

    def reset(self) -> np.ndarray:
        self.curr_step_idx = 0
        self.curr_v = np.random.uniform(0.0,
                                        1.0,
                                        (self.n_params,)).astype(np.float32)
        return self.curr_v

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == '__main__':
    train_env = TestEnv()
    # train_env = make_vec_env(TestEnv, n_envs=4, vec_env_cls=DummyVecEnv)
    # train_env = make_vec_env(TestEnv, n_envs=4, vec_env_cls=SubprocVecEnv)
    # env = make_vec_env(TestEnv, n_envs=4, vec_env_cls=SubprocVecEnv)

    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[4])
    policy_kwargs = dict(net_arch=[64, 64])
    # policy_kwargs = dict(net_arch=[dict(vf=[8], pi=[8])])
    model = PPO('MlpPolicy', train_env, policy_kwargs=policy_kwargs, verbose=1)
    # model = A2C('MlpPolicy', train_env, policy_kwargs=policy_kwargs, verbose=1)
    # model = PPO('MlpPolicy',
    #             env,
    #             verbose=1)
    # model = A2C('MlpPolicy', env, verbose=1)
    # model = SAC('MlpPolicy', env, verbose=1)
    # model = TD3('MlpPolicy', env, verbose=1)
    n_train_steps = 10000
    n_render_steps = 1000
    n_done = 0
    n_stuck = 0
    curr_step_idx = 0
    done_steps = []
    stuck_steps = []

    model.learn(total_timesteps=n_train_steps)
    # model.save('testing_model')

    render_env = TestEnv()
    log.info('Rendering')
    obs = render_env.reset()
    for i in range(n_render_steps):
        action, _states = model.predict(obs, deterministic=True)
        curr_step_idx += 1
        prev_obs = obs
        obs, reward, done, info = render_env.step(action)
        log.debug(f'render prev_obs = {prev_obs}')
        log.debug(f'render action = {action}')
        log.debug(f'render obs = {obs}')
        log.debug(f'render reward = {reward}')

        if done:
            if not info['stuck']:
                log.debug('done!')
                obs = render_env.reset()
                log.debug(f'env reset to {obs}')
                prev_obs = obs
                n_done += 1
                done_steps.append(curr_step_idx)
                curr_step_idx = 0
            else:
                log.debug('stuck!')
                obs = render_env.reset()
                log.debug(f'env reset to {obs}')
                prev_obs = obs
                n_stuck += 1
                stuck_steps.append(curr_step_idx)
                curr_step_idx = 0

        log.debug('')

    render_env.close()
    log.info(f'n_train_steps = {n_train_steps}')
    log.info(f'n_render_steps = {n_render_steps}')
    log.info(f'n_done = {n_done}')
    log.debug(f'len done_steps = {len(done_steps)}')
    if done_steps:
        log.info(f'mean done_steps = {np.mean(done_steps):.4f}')
    log.debug(f'sum done_steps = {sum(done_steps)}')

    log.info(f'n_stuck = {n_stuck}')
    log.debug(f'len stuck_steps = {len(stuck_steps)}')
    if stuck_steps:
        log.info(f'mean stuck_steps = {np.mean(stuck_steps):.4f}')
    log.debug(f'sum stuck_steps = {sum(stuck_steps)}')

    log.info(f'ending curr_step_idx = {curr_step_idx}')
    log.info(f'step sum = {sum(done_steps) + sum(stuck_steps) + curr_step_idx}')
