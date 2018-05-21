"""
Environments and wrappers for Sonic training.
"""

import numpy as np
import os

import gym
import gym_remote.client as grc

from baselines.common.atari_wrappers import WarpFrame, FrameStack

from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subprocess_vec_env import SubprocessVecEnv
def get_env():
    return env

def make_env(stack=True, extra_wrap_fn=None):
    if 'RETRO_RECORD' in os.environ:
      from retro_contest.local import make
      game=os.environ['RETRO_GAME']
      state=os.environ['RETRO_STATE']
      env_id = game + "-" + state
      env = make(game=game, state=state, bk2dir=os.environ['RETRO_RECORD'])
    else:
      env_id = 'tmp/sock'
      env = grc.RemoteEnv('tmp/sock')

    env = SonicDiscretizer(env)
    env = WarpFrame(env)
    if stack:
      env = FrameStack(env, 4)
    if extra_wrap_fn is not None:
      env = extra_wrap_fn(env)
    return env_id, env

def make_vec_env(stack=True, extra_wrap_fn=None):
    def wrap_env(env):
      env = SonicDiscretizer(env)
      env = WarpFrame(env)
      if stack:
        env = FrameStack(env, 4)
      if extra_wrap_fn is not None:
        env = extra_wrap_fn(env)
      return env
 
    if 'RETRO_RECORD' in os.environ:
      from retro_contest.local import make
      record_dir=os.environ['RETRO_RECORD']
      def prepare_env(game, state):
        bk2dir = record_dir + "/" + game + "-" + state
        os.mkdir(bk2dir)
        return wrap_env(make(game=game, state=state, bk2dir=bk2dir))
      game=os.environ['RETRO_GAME']
      state=os.environ['RETRO_STATE']
      return SubprocessVecEnv([(game + "-" + state, lambda: prepare_env(game, state))])
    else:
      return DummyVecEnv([('tmp/sock', lambda: wrap_env(grc.RemoteEnv('tmp/sock')))])

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

