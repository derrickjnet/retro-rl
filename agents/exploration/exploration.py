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

class Exploration:
   def __init__(self, env_idx, env_id, allow_backtracking=True, max_exploration_steps=None, log_file=sys.stdout, save_state_dir=None):
     self.env_idx = env_idx
     self.env_id = env_id 
     self.log_file = log_file
     self.save_state_dir = save_state_dir
     self.allow_backtracking = True
     self.max_exploration_steps = max_exploration_steps
     self.episode=None
     self.global_reward = 0
     self.total_steps=0
     self.episode_step = 0
     self.total_reward = 0
     self.total_adjusted_reward = 0
     self.total_extra_reward = 0
     self.global_visited = dict()

   def reset(self, obs):
     if self.episode is None:
       self.episode = 0
     else:
       print("EPISODE: timestamp=%s env_idx=%s env_id=%s total_steps=%s episode=%s episode_step=%s total_reward=%s total_adjusted_reward=%s total_extra_reward=%s avg_episode_reward=%s" % (datetime.datetime.now(), self.env_idx, self.env_id, self.total_steps, self.episode, self.episode_step, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, self.global_reward / (1+float(self.episode))), file=self.log_file)
       sys.stdout.flush()
       self.episode += 1
     self.local_visited = dict()
     self.episode_step = 0
     self.total_reward = 0
     self.total_adjusted_reward = 0
     self.total_extra_reward = 0
     self.last_obs = None
     self.last_info = None
  
   def action_meta(self, action_meta):
     print("%s: timestmap=%s %s" % (action_meta[0], datetime.datetime.now(), action_meta[1]), file=self.log_file)

   def step(self, action, obs, reward, done, info, state_embedding, state_embedding_reward): 
     if 'initial_reward' in info:
       assert self.episode_step == 0
       initial_reward = info['initial_reward']
       self.total_reward += initial_reward
       self.total_adjusted_reward += initial_reward
       self.global_reward += initial_reward

     self.episode_step += 1
     self.total_steps += 1
     self.total_reward += reward
     self.global_reward += reward
     adjusted_reward = max(0, self.total_reward - self.total_adjusted_reward)
     self.total_adjusted_reward = max(self.total_adjusted_reward, self.total_reward)

     if state_embedding is not None:
       state_encoding = tuple(np.round(state_embedding).tolist())
       state_embedding = tuple(state_embedding.tolist())
     else:
       state_encoding = None

     cell_x = int(self.total_reward/100.0)
     cell_y = int(info.get('y',0)/10.0)
     cell_key = (cell_x, cell_y, state_encoding)

     self.local_visited[cell_key] = self.local_visited.get(cell_key, 0) + 1
     self.global_visited[cell_key] = self.global_visited.get(cell_key, 0) + 1

     relative_x = self.total_reward / 9000.0

     if 'y' in info:
       relative_y = info['y'] / info.get('screen_x_end',1.0)
       extra_reward_weight = 10.0 * math.sqrt(relative_x**2 + relative_y**2)
     else:
       relative_y = None
       extra_reward_weight = 10.0 * relative_x

     exploration_local_reward = extra_reward_weight / self.local_visited[cell_key]
     exploration_global_reward = extra_reward_weight / math.sqrt(self.global_visited[cell_key])
 
     extra_reward = exploration_local_reward + exploration_global_reward + state_embedding_reward
     if self.max_exploration_steps != None:
       extra_reward_scale = max(0, self.max_exploration_steps - self.total_steps) / float(self.max_exploration_steps)
     else:
       extra_reward_scale = 1.0
     extra_reward *= extra_reward_scale
     self.total_extra_reward += extra_reward  

     if self.allow_backtracking:
       final_reward = adjusted_reward + extra_reward
     else:
       final_reward = reward + extra_reward

     timestamp = datetime.datetime.now()
     print("EXPLORE: timestamp=%s env_idx=%s env_id=%s total_steps=%s episode=%s episode_step=%s local_visited=%s global_visited=%s extra_reward_weight=%s extra_reward_scale=%s exploration_local_reward=%s exploration_global_reward=%s state_embedding_reward=%s relative_x=%s relative_y=%s cell=%s encoding=%s embedding=%s" % (timestamp, self.env_idx, self.env_id, self.total_steps, self.episode, self.episode_step, self.local_visited[cell_key], self.global_visited[cell_key], extra_reward_weight, extra_reward_scale, exploration_local_reward, exploration_global_reward, state_embedding_reward, relative_x, relative_y, cell_key, state_encoding, state_embedding), file=self.log_file)
     print("STEP: timestamp=%s env_idx=%s env_id=%s total_steps=%s episode=%s episode_step=%s action=%s done=%s, reward=%s adjusted_reward=%s extra_reward=%s current_reward=%s current_adjusted_reward=%s current_extra_reward=%s info=%s" % (timestamp, self.env_idx, self.env_id, self.total_steps, self.episode, self.episode_step, action, done, reward, adjusted_reward, extra_reward, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, info), file=self.log_file)
     sys.stdout.flush()

     if self.save_state_dir is not None:
       event = {
         'timestamp' : timestamp,
         'env_idx' : self.env_idx,
         'env_id' : self.env_id,
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
       save_state_dir_name = self.save_state_dir + "/" + self.env_id + "-" + str(self.episode).zfill(4) + ".steps/"
       if self.episode_step == 1: 
         os.mkdir(save_state_dir_name) 
       save_step_file_name = save_state_dir_name + str(self.episode) + "." + str(self.episode_step) + ".step.gz"
       save_step_file = gzip.open(save_step_file_name, "wb")
       cloudpickle.dump(event, save_step_file)
       save_step_file.close() 

     self.last_obs = obs
     self.last_info = info
     return final_reward

   def close(self):
     pass
