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
   def __init__(self, env, max_exploration_steps=None, state_encoder=None):
     super(ExplorationEnv, self).__init__(env)
     self.max_exploration_steps = max_exploration_steps
     self.state_encoder = state_encoder
     self.episode=None
     self.total_steps=0
     self.episode_step = 0
     self.total_reward = 0
     self.total_adjusted_reward = 0
     self.total_extra_reward = 0
     self.global_visited = dict()
 
   def get_movie_id(self):
     env = self.env
     while hasattr(env, 'env'):
       env = env.env
     return env.movie_id - 1

   def reset(self):
     print("EPISODE: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s total_reward=%s total_adjusted_reward=%s total_extra_reward=%s" % (datetime.datetime.now(), self.get_movie_id(), self.total_steps, self.episode, self.episode_step, self.total_reward, self.total_adjusted_reward, self.total_extra_reward))
     sys.stdout.flush()
     if self.episode is None:
       self.episode = 0
     else:
       self.episode += 1
     self.visited = dict()
     self.episode_step = 0
     self.total_reward = 0
     self.total_adjusted_reward = 0
     self.total_extra_reward = 0
     self.last_obs = None
     self.last_info = None
     return super().reset()

   def step(self, action): 
     self.episode_step += 1
     self.total_steps += 1
     obs, reward, done, info = super().step(action) 
     self.total_reward += reward
     adjusted_reward = max(0, self.total_reward - self.total_adjusted_reward)
     self.total_adjusted_reward = max(self.total_adjusted_reward, self.total_reward)
     if self.state_encoder is not None:
       state_embedding_start = datetime.datetime.now()
       state_embedding = list(self.state_encoder.encode(obs))
       print("STATE_EMBEDDING: runtime=%s sec" % ((datetime.datetime.now() - state_embedding_start).total_seconds(),))
       state_encoding = tuple(map(lambda v: round(v), state_embedding))
     else:
       state_embedding = [int(info['y']/100.0)]
       state_encoding = state_embedding
     cell_key = (int(self.total_reward/100), state_encoding)
     self.visited[cell_key] = self.visited.get(cell_key, 0) + 1
     self.global_visited[cell_key] = self.global_visited.get(cell_key, 0) + 1
     extra_reward = 0.0
     extra_reward += 10 * self.total_reward / 9000.0 / self.visited[cell_key]
     #extra_reward += 10 * self.total_reward / 9000.0 / math.sqrt(self.global_visited[cell_key])
     if self.max_exploration_steps != None:
       extra_reward *= max(0, self.max_exploration_steps - self.total_steps) / float(self.max_exploration_steps)
     self.total_extra_reward += extra_reward
     timestamp = datetime.datetime.now()
     print("CELL: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s visited=%s global_visited=%s cell=%s embedding=%s" % (timestamp, self.get_movie_id(), self.total_steps, self.episode, self.episode_step, self.visited[cell_key], self.global_visited[cell_key], cell_key, state_embedding))
     print("STEP: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s action=%s reward=%s adjusted_reward=%s extra_reward=%s current_reward=%s current_adjusted_reward=%s current_extra_reward=%s info=%s" % (timestamp, self.get_movie_id(), self.total_steps, self.episode, self.episode_step, action, reward, adjusted_reward, extra_reward, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, info))
     sys.stdout.flush()
     if 'RETRO_RECORD' in os.environ:
       record = {
         'timestamp' : timestamp,
         'movie_id' : self.get_movie_id(),
         'total_steps' : self.total_steps, 
         'episode' : self.episode,
         'episode_step' : self.episode_step,
         #'last_obs' : self.last_obs,
         'last_info' : self.last_info,
         'action' : action,
         'obs' : obs,
         'reward' : reward,
         'adjusted_reward' : adjusted_reward,
         'extra_reward' : extra_reward,
         'done' : done,
         'info' : info,
         'total_reward' : self.total_reward,
         'total_adjusted_reward' : self.total_adjusted_reward,
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
     return obs, adjusted_reward + extra_reward, done, info

