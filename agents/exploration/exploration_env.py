import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc

import os
import sys
import math
import datetime
import cloudpickle
import gzip

class ExplorationEnv(gym.Wrapper):
   def __init__(self, env_id, env, exploration_f, state_encoder=None, record_dir=os.environ['RETRO_RECORD']):
     super(ExplorationEnv, self).__init__(env)
     self.env_id = env_id
     self.env = env
     self.state_encoder = state_encoder
     self.exploration = exploration_f(env_id, log_file=sys.stdout, save_state_dir=record_dir) 

   def reset(self):
     obs = super().reset()
     exploration.reset(obs)
     return obs

   def step(self, action): 
     obs, reward, done, info = super().step(action)
     final_reward = exploration.step(action, obs, rew, done, info)
     return obs, final_reward, done, info

