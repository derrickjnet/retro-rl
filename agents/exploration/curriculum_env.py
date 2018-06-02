import sys
import os
import random
import math

import gym
import retro

class CurriculumEnv(gym.Wrapper):
   def __init__(self, env, unwrapped_env, movie_path, log_file=sys.stdout, max_curriculum_steps=int(os.environ.get("RETRO_CURRICULUM_MAX_STEPS", 0))):
     super(CurriculumEnv, self).__init__(env)
     self.unwrapped_env = unwrapped_env
     self.movie_path = movie_path
     self.log_file = log_file
     self.max_curriculum_steps = max_curriculum_steps
     self.movie_length = self.compute_movie_length()
     
     print("CURRICULUM_MOVIE: length=%s path=%s" % (self.movie_length, movie_path,), file=self.log_file)
     self.log_file.flush()
     
     self.total_steps = 0

   def _replay_reset(self):
     movie = retro.Movie(self.movie_path)
     movie.step()
     self.unwrapped_env.initial_state = movie.get_state() 
     obs = super().reset()
     return movie, obs

   def _replay_step(self, movie):
     action_keys = []
     for i in range(self.unwrapped_env.NUM_BUTTONS):
       action_keys.append(movie.get_key(i))
     obs, reward, done, info = self.unwrapped_env.step(action_keys)
     if info['screen_x'] == info['screen_x_end']:
        done = True
     return obs, reward, done, info, action_keys 

   def compute_movie_length(self):
     movie = self._replay_reset()[0]
     done = False
     movie_length = 0
     while not done and movie.step():
       done = self._replay_step(movie)[2]
       movie_length += 1
     return movie_length

   def reset(self):
     curriculum_progress = min(1.0,self.total_steps / max(1.0,float(self.max_curriculum_steps)))
     print("CURRICULUM_RESET: total_steps=%s curriculum_progress=%s" % (self.total_steps, curriculum_progress), file=self.log_file)
     self.log_file.flush()

     success = False
     attempt = 0
     while not success:
       attempt += 1
       replay_length = math.ceil(random.random() * (1-curriculum_progress) * self.movie_length)
       print("CURRICULUM_REPLAY_START: total_steps=%s attempt=%s replay_length=%s" % (self.total_steps, attempt, replay_length), file=self.log_file)

       movie, obs = self._replay_reset()

       done = False
       replay_steps = 0
       replay_reward = 0
       while not done and replay_steps < replay_length and movie.step():
         replay_steps += 1
         obs, reward, done, info, action_keys = self._replay_step(movie)
         print("CURRICULUM_STEP: total_steps=%s attempt=%s replay_steps=%s/%s action_keys=%s reward=%s replay_reward=%s done=%s info=%s" % (self.total_steps, attempt, replay_steps, replay_length, action_keys, reward, replay_reward, done, info), file=self.log_file)
         replay_reward += reward
       
       if not done:
         success = True
         self.replay_reward = replay_reward

     print("CURRICULUM_REPLAY_END: total_steps=%s attempt=%s replay_steps=%s/%s replay_reward=%s" % (self.total_steps, attempt, replay_steps, replay_length, replay_reward), file=self.log_file)
     self.log_file.flush()
     return obs

   def step(self, action):
     self.total_steps += 1
     obs, reward, done, info = super().step(action)
     if self.replay_reward:
       info['initial_reward'] = self.replay_reward
       self.replay_reward = None
     return obs, reward, done, info 
