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
   def __init__(self, env):
     super(ExplorationEnv, self).__init__(env)
     self.episode=0
     self.total_steps=0
     self.max_visited_x = None
     self.max_x = None
     self.episode_step = 0
     self.total_reward = 0
     self.total_extra_reward = 0
 
   def get_movie_id(self):
     env = self.env
     while hasattr(env, 'env'):
       env = env.env
     return env.movie_id

   def reset(self):
     print("EPISODE: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s max_visisted_x=%s max_x=%s total_reward=%s total_extra_reward=%s" % (datetime.datetime.now(), self.get_movie_id(), self.total_steps, self.episode, self.episode_step, self.max_visited_x, self.max_x, self.total_reward, self.total_extra_reward))
     sys.stdout.flush()
     self.episode += 1
     self.max_visited_x = None
     self.max_x = None
     self.visited = dict()
     self.episode_step = 0
     self.total_reward = 0
     self.total_extra_reward = 0
     self.last_obs = None
     self.last_info = None
     return super().reset()

   def step(self, action): 
     obs, reward, done, info = super().step(action) 
     if self.max_x is None:
       self.max_x = info['screen_x_end']
     current_x = info['x']
     current_y = info['y']
     if self.max_visited_x is not None: 
       self.max_visited_x = max(self.max_visited_x, current_x)
     else:
       self.max_visited_x = current_x
     cell_size = self.max_x / 10
     cell_x = int(current_x / cell_size)
     cell_y = int(current_y / cell_size)
     cell_key = (cell_x, cell_y)
     if cell_key not in self.visited:
       self.visited[cell_key] = 1
     else:
       self.visited[cell_key] += 1
     extra_reward = math.sqrt(current_x**2 + current_y**2) * 9000.0 / self.max_x / (self.max_x / cell_size) / self.visited[cell_key]
     self.total_steps += 1
     self.episode_step += 1
     self.total_reward += reward
     self.total_extra_reward += extra_reward
     timestamp = datetime.datetime.now()
     print("STEP: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s action=%s reward=%s extra_reward=%s info=%s" % (timestamp, self.get_movie_id(), self.total_steps, self.episode, self.episode_step, action, reward, extra_reward, info))
     sys.stdout.flush()
     if 'RETRO_RECORD' in os.environ:
       record = {
         'timestamp' : timestamp,
         'movie_id' : self.get_movie_id(),
         'total_steps' : self.total_steps, 
         'episode' : self.episode,
         'episode_step' : self.episode_step,
         'last_obs' : self.last_obs,
         'last_info' : self.last_info,
         'action' : action,
         'obs' : obs,
         'reward' : reward,
         'done' : done,
         'extra_reward' : extra_reward,
         'info' : info,
         'total_reward' : self.total_reward,
         'total_extra_reward' : self.total_extra_reward
       }
       record_dir_name = os.environ['RETRO_RECORD'] + "/" + os.environ['RETRO_GAME'] + "-" + os.environ['RETRO_STATE'] + "-" + str(self.get_movie_id()).zfill(4) + ".steps/"
       if self.episode_step == 1: 
         os.mkdir(record_dir_name) 
       record_file_name = record_dir_name + str(self.episode) + "." + str(self.episode_step) + ".step.gz"
       record_file = gzip.open(record_file_name, "wb")
       cloudpickle.dump(record, record_file)
       record_file.close() 

     self.last_obs = obs
     self.last_info = info
     return obs, reward + extra_reward, done, info

