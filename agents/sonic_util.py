"""
Environments and wrappers for Sonic training.
"""

import numpy as np
import os
import csv

import gym
import gym_remote.client as grc
from batched_env.batched_gym_env import batched_gym_env, BatchedGymEnv

from baselines.common.atari_wrappers import WarpFrame

from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subprocess_vec_env import SubprocessVecEnv

from exploration.curriculum_env import CurriculumEnv

def get_env():
    return env

def wrap_env(env, extra_wrap_fn):
  env = SonicDiscretizer(env)
  env = WarpFrame(env)
  if extra_wrap_fn is not None:
    env = extra_wrap_fn(env)
  return env

def make_env(extra_wrap_fn=None):
  if 'RETRO_RECORD' in os.environ:
    from retro_contest.local import make
    game=os.environ['RETRO_GAME']
    state=os.environ['RETRO_STATE']
    env_id = game + "-" + state
    env = make(game=game, state=state, bk2dir=os.environ['RETRO_RECORD'])
  else:
    env_id = 'tmp/sock'
    env = grc.RemoteEnv('tmp/sock')
  return env_id, wrap_env(env)

def build_envs(extra_wrap_fn=None):
  root_dir = os.environ['RETRO_ROOT_DIR']
  movie_path_prefix =  os.environ.get('RETRO_CURRICULUM_MOVIE_PATH_PREFIX')
  def compose_env(game, state, bk2dir):
    env = make(game=game, state=state, bk2dir=bk2dir)
    if movie_path_prefix is not None:
        log_path=root_dir + "/" + game + "-" + state + "/"
        movie_path = movie_path_prefix + "/" + game + "/contest/" + game + "-" + state + "-0000.bk2"
        print("CURRICULUM_LOG: %s" % (log_path,))
        env = CurriculumEnv(env, env.env.env, movie_path=movie_path, log_path=log_path)
    return wrap_env(env, extra_wrap_fn)
  from retro_contest.local import make
  if 'RETRO_RECORD_DIR' in os.environ:
    record_dir=os.environ['RETRO_RECORD_DIR']
    def build_env(game, state):
      bk2dir = record_dir + "/" + game + "-" + state
      os.makedirs(bk2dir, exist_ok=True)
      return lambda: compose_env(game, state, bk2dir)
  else:
    def build_env(game, state):
      return lambda: compose_env(game, state, None)
  subenv_ids = []
  subenvs = []
  if 'RETRO_GAMESFILE' in os.environ:
    for row in csv.DictReader(open(os.environ['RETRO_GAMESFILE'], 'r')):
      game = row['game']
      state = row['state']
      subenv_ids.append(game + "-" + state)
      subenvs.append(build_env(game, state))
  else:
    game=os.environ['RETRO_GAME']
    state=os.environ['RETRO_STATE']
    subenv_ids.append(game + "-" + state)
    subenvs.append(build_env(game, state))
  return (subenv_ids, subenvs)

def make_batched_env(extra_wrap_fn=None):
  if 'RETRO_ROOT_DIR' in os.environ:
    subenv_ids, subenvs = build_envs(extra_wrap_fn=extra_wrap_fn)
    env = batched_gym_env(subenvs, sync=False)
    #env = BatchedGymEnv([[subenv() for subenv in subenvs]])
    env.env_ids = subenv_ids
    return env
  else:
    env = BatchedGymEnv([[wrap_env(grc.RemoteEnv('tmp/sock'), extra_wrap_fn=extra_wrap_fn)]])
    env.env_ids = ['tmp/sock']
    return env

def make_vec_env(extra_wrap_fn=None):
  if 'RETRO_ROOT_DIR' in os.environ:
    subenv_ids, subenvs = build_envs(extra_wrap_fn=extra_wrap_fn)
    return SubprocessVecEnv(zip(subenv_ids, subenvs))
  else:
    return DummyVecEnv([('tmp/sock', lambda: wrap_env(grc.RemoteEnv('tmp/sock'), extra_wrap_fn=extra_wrap_fn))])

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

