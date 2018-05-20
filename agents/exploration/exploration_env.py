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
   def __init__(self, env, allow_backtracking=True, max_exploration_steps=None, state_encoder=None):
     super(ExplorationEnv, self).__init__(env)
     self.allow_backtracking = True
     self.max_exploration_steps = max_exploration_steps
     self.state_encoder = state_encoder
     self.episode=None
     self.global_reward = 0
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
     if self.episode is None:
       self.episode = 0
     else:
       self.episode += 1
     if self.episode > 0:
       print("EPISODE: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s total_reward=%s total_adjusted_reward=%s total_extra_reward=%s avg_episode_reward=%s" % (datetime.datetime.now(), self.get_movie_id(), self.total_steps, self.episode, self.episode_step, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, self.global_reward / float(self.episode)))
       sys.stdout.flush()
     self.local_visited = dict()
     self.episode_step = 0
     self.episode_rings = 0
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
     self.global_reward += reward
     adjusted_reward = max(0, self.total_reward - self.total_adjusted_reward)
     self.total_adjusted_reward = max(self.total_adjusted_reward, self.total_reward)

     if self.state_encoder is not None:
       state_embedding_start = datetime.datetime.now()
       state_embedding = list(self.state_encoder.encode(np.expand_dims(np.expand_dims(obs[:,:,-1],-1),0)))[0]
       print("STATE_EMBEDDING: runtime=%s sec" % ((datetime.datetime.now() - state_embedding_start).total_seconds(),))
       state_encoding = tuple(map(lambda v: round(v), state_embedding))
     else:
       state_embedding = []
       state_encoding = [] 

     cell_x = int(self.total_reward/100.0)
     cell_y = int(info.get('y',0)/10.0)
     cell_key = (cell_x, cell_y, state_encoding)

     self.local_visited[cell_key] = self.local_visited.get(cell_key, 0) + 1
     self.global_visited[cell_key] = self.global_visited.get(cell_key, 0) + 1

     relative_x = self.total_reward / 9000.0

     if 'y' in info:
       relative_y = info['y'] / info.get('screen_x_end',1.0)
       exploration_reward_weight = 10.0 * math.sqrt(relative_x**2 + relative_y**2)
     else:
       relative_y = None
       exploration_reward_weight = 10.0 * relative_x

     exploration_local_reward = exploration_reward_weight / self.local_visited[cell_key]
     exploration_global_reward = exploration_reward_weight / math.sqrt(self.global_visited[cell_key])
  
     exploration_rings_weight = 0 
     if 'rings' in info:
       new_episode_rings = info.get('rings',0)
       extra_rings_reward = exploration_rings_weight * max(0, new_episode_rings - self.episode_rings)
       self.episode_rings = new_episode_rings
     else:
       extra_rings_reward = 0.0

     extra_reward = exploration_local_reward + exploration_global_reward + extra_rings_reward
     if self.max_exploration_steps != None:
       extra_reward_scale = max(0, self.max_exploration_steps - self.total_steps) / float(self.max_exploration_steps)
     else:
       extra_reward_scale = 1.0

     self.total_extra_reward += extra_reward_scale * extra_reward

     if self.allow_backtracking:
       final_reward = adjusted_reward + extra_reward
     else:
       final_reward = reward + extra_reward

     timestamp = datetime.datetime.now()
     print("EXPLORE: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s local_visited=%s global_visited=%s exploration_reward_weight=%s extra_reward_scale=%s exploration_local_reward=%s exploration_global_reward=%s extra_rings_reward=%s relative_x=%s relative_y=%s cell=%s embedding=%s" % (timestamp, self.get_movie_id(), self.total_steps, self.episode, self.episode_step, self.local_visited[cell_key], self.global_visited[cell_key], exploration_reward_weight, extra_reward_scale, exploration_local_reward, exploration_global_reward, extra_rings_reward, relative_x, relative_y, cell_key, state_embedding.tolist()))
     print("STEP: timestamp=%s movie_id=%s total_steps=%s episode=%s episode_step=%s action=%s reward=%s adjusted_reward=%s extra_reward=%s current_reward=%s current_adjusted_reward=%s current_extra_reward=%s info=%s" % (timestamp, self.get_movie_id(), self.total_steps, self.episode, self.episode_step, action, reward, adjusted_reward, extra_reward, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, info))
     sys.stdout.flush()

     if False and 'RETRO_RECORD' in os.environ:
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
     return obs, final_reward, done, info

