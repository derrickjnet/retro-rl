import gym
import numpy as np

import os
import sys
import math

class ExplorationEnv(gym.Wrapper):
   def __init__(self, env_id, env, exploration_f, state_encoder=None, record_dir=os.environ['RETRO_RECORD']):
     super(ExplorationEnv, self).__init__(env)
     self.env_id = env_id
     self.env = env
     self.state_encoder = state_encoder
     if record_dir is not None:
        self.log_file = sys.stdout
     else:
        self.log_file = open("/dev/null", "w")
     self.exploration = exploration_f(env_id, log_file=self.log_file, save_state_dir=record_dir) 

   def reset(self):
     obs = super().reset()
     self.exploration.reset(obs)
     return obs

   def action_meta(self, action_meta):
     exploration.action_meta(action_meta)
   
   def step(self, action): 
     obs, rew, done, info = super().step(action)
     if self.state_encoder is not None:
       state_encodings, state_encoding_weights = self.state_encoder.encode(np.expand_dims(np.expand_dims(obs[:,:,-1],0),-1))
       state_encoding = list(state_encodings)[0]
       state_encoding_weight = state_encoding_weights[0]
     else:
       state_encoding = None
       state_encoding_rewards = 0 
     final_reward = self.exploration.step(action, obs, rew, done, info, state_encoding, state_encoding_weight)
     return obs, final_reward, done, info

   def close(self):
     self.exploration.close()
