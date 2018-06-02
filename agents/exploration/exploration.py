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
   def __init__(self, env_idx, env_id, allow_backtracking=True, max_exploration_steps=os.environ.get("RETRO_MAX_EXPLORATION_STEPS",None), extra_local_weight=float(os.environ.get("RETRO_EXPLORATION_LOCAL_WEIGHT", 10.0)), extra_global_weight=float(os.environ.get("RETRO_EXPLORATION_GLOBAL_WEIGHT", 10.0)), extra_predictor_weight=float(os.environ.get("RETRO_EXPLORATION_PREDICTOR_WEIGHT", 10.0)), log_file=sys.stdout, save_state_dir=None):
     self.env_idx = env_idx
     self.env_id = env_id 
     self.log_file = log_file
     self.save_state_dir = save_state_dir
     self.allow_backtracking = True
     self.max_exploration_steps = int(max_exploration_steps) if max_exploration_steps else None
     self.extra_local_weight = extra_local_weight
     self.extra_global_weight = extra_global_weight
     self.extra_predictor_weight = extra_predictor_weight
     self.episode=None
     self.cumulative_reward = 0
     self.cumulative_reward_emav = 0
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
       self.episode += 1
     if self.episode > 0:
       self.cumulative_reward += self.total_reward
       self.cumulative_reward_emav = 0.9 * self.cumulative_reward_emav + 0.1 * self.cumulative_reward
       print("EPISODE: timestamp=%s env_idx=%s env_id=%s total_steps=%s episode=%s episode_step=%s total_reward=%s total_adjusted_reward=%s total_extra_reward=%s avg_episode_reward=%s emav_episode_reward=%s" % (datetime.datetime.now(), self.env_idx, self.env_id, self.total_steps, self.episode, self.episode_step, self.total_reward, self.total_adjusted_reward, self.total_extra_reward, self.cumulative_reward / float(self.episode), self.cumulative_reward_emav), file=self.log_file)
       sys.stdout.flush()
     self.local_visited = dict()
     self.episode_step = 0
     self.total_reward = 0
     self.total_adjusted_reward = 0
     self.total_extra_reward = 0
     self.last_obs = None
     self.last_info = None
  
   def action_meta(self, action_meta):
     print("%s: timestmap=%s %s" % (action_meta[0], datetime.datetime.now(), action_meta[1]), file=self.log_file)

   def step(self, action, obs, reward, done, info, state_embedding, state_predictor_reward): 
     self.episode_step += 1
     self.total_steps += 1
     self.total_reward += reward
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
       visitation_reward_weight = math.sqrt(relative_x**2 + relative_y**2)
     else:
       relative_y = None
       visitation_reward_weight = relative_x

     visitation_local_reward = self.extra_local_weight * visitation_reward_weight / self.local_visited[cell_key]
     visitation_global_reward = self.extra_global_weight * visitation_reward_weight / math.sqrt(self.global_visited[cell_key])

     predictor_reward = self.extra_predictor_weight * state_predictor_reward
 
     extra_reward = visitation_local_reward + visitation_global_reward + predictor_reward 

     if self.max_exploration_steps != None:
       extra_reward_scale = max(0, self.max_exploration_steps - self.total_steps) / max(1.0,float(self.max_exploration_steps))
     else:
       extra_reward_scale = 1.0
     extra_reward *= extra_reward_scale
     self.total_extra_reward += extra_reward  

     if self.allow_backtracking:
       final_reward = adjusted_reward + extra_reward
     else:
       final_reward = reward + extra_reward

     timestamp = datetime.datetime.now()
     print("EXPLORE: timestamp=%s env_idx=%s env_id=%s total_steps=%s episode=%s episode_step=%s local_visited=%s global_visited=%s visitation_reward_weight=%s visitation_local_reward=%s visitation_global_reward=%s predictor_reward=%s extra_reward_scale=%s relative_x=%s relative_y=%s cell=%s encoding=%s embedding=%s" % (timestamp, self.env_idx, self.env_id, self.total_steps, self.episode, self.episode_step, self.local_visited[cell_key], self.global_visited[cell_key], visitation_reward_weight, visitation_local_reward, visitation_global_reward, predictor_reward, extra_reward_scale, relative_x, relative_y, cell_key, state_encoding, state_embedding), file=self.log_file)
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
