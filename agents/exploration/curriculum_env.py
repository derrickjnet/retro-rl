import sys
import os
import random
import math

import gym
import retro

class CurriculumEnv(gym.Wrapper):
   def __init__(self, env, unwrapped_env, movie_path, log_file=sys.stdout, randomize_curriculum=os.environ.get("RETRO_CURRICULUM_RANDOMIZE","false") == "true", max_curriculum_steps=int(os.environ.get("RETRO_CURRICULUM_MAX_STEPS", 0))):
     super(CurriculumEnv, self).__init__(env)
     self.unwrapped_env = unwrapped_env
     self.movie_path = movie_path
     self.log_file = log_file
     self.max_curriculum_steps = max_curriculum_steps
     self.movie_length = self.compute_movie_length()
     self.randomize_curriculum = randomize_curriculum
     
     print("CURRICULUM_MOVIE: length=%s path=%s" % (self.movie_length, movie_path,), file=self.log_file)
     self.log_file.flush()

     self.episode = -1 
     self.total_steps = 0
     self.total_replay_steps = 0

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
     return obs, reward, done, info, action_keys 

   def compute_movie_length(self):
     movie = self._replay_reset()[0]
     done = False
     movie_length = 0
     while movie.step():
       movie_length += 1
       _, _, done, info, _ = self._replay_step(movie)
       if done or info['screen_x'] == info['screen_x_end']:
         break
     return movie_length

   def reset(self):
     self.episode += 1
     success = False
     attempt = 0
     while not success:
       attempt += 1

       try:
         curriculum_progress = min(1.0,self.total_steps / max(1.0,float(self.max_curriculum_steps)))
         if self.randomize_curriculum:
           replay_length = max(0,min(self.movie_length, math.floor(random.random() * (1-curriculum_progress) * self.movie_length)))
         else:
           replay_length = max(0,min(self.movie_length, math.floor((1-curriculum_progress) * self.movie_length)))
           replay_length = math.floor(replay_length * 0.99 ** attempt)

         print("CURRICULUM_REPLAY_BEGIN: total_steps=%s total_replay_steps=%s episode=%s attempt=%s curriculum_progress=%s movie_length=%s replay_length=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, curriculum_progress, self.movie_length, replay_length), file=self.log_file)

         movie, obs = self._replay_reset()

         done = False
         replay_steps = 0
         replay_reward = 0
         while not done and replay_steps < replay_length and movie.step():
           self.total_replay_steps += 1
           replay_steps += 1
           obs, reward, done, info, action_keys = self._replay_step(movie)
           print("CURRICULUM_STEP: total_steps=%s total_replay_steps=%s episode=%s attempt=%s movie_length=%s replay_length=%s replay_steps=%s action_keys=%s reward=%s replay_reward=%s done=%s info=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, self.movie_length, replay_length, replay_steps, action_keys, reward, replay_reward, done, info), file=self.log_file)
           replay_reward += reward
       
         if not done:
           success = True
           self.replay_reward = replay_reward
       except RuntimeError as e:
         import traceback
         traceback.print_exc()
  
     print("CURRICULUM_REPLAY_END: total_steps=%s total_replay_steps=%s episode=%s attempt=%s movie_length=%s replay_length=%s replay_reward=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, self.movie_length, replay_length, replay_reward), file=self.log_file)
     self.log_file.flush()
     return obs

   def step(self, action):
     self.total_steps += 1
     try:
       obs, reward, done, info = super().step(action)
     except RuntimeError as e:
       import traceback
       traceback.print_exc()
       raise
     if self.replay_reward:
       info['initial_reward'] = self.replay_reward
       self.replay_reward = None
     return obs, reward, done, info 
