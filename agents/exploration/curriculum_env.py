import sys
import os
import random
import math
import copy
import cloudpickle

import gym
import retro

class CurriculumEnv(gym.Wrapper):
   def __init__(self, env, movie_path, log_file=sys.stdout, randomize_curriculum=os.environ.get("RETRO_CURRICULUM_RANDOMIZE") == "true", max_curriculum_steps=int(os.environ.get("RETRO_CURRICULUM_MAX_STEPS", 0))):
     super(CurriculumEnv, self).__init__(env)
     self.movie_path = movie_path
     self.log_file = log_file
     self.randomize_curriculum = randomize_curriculum
     self.max_curriculum_steps = max_curriculum_steps
     self.movie_length, self.movie_states, self.movie_rewards, self.movie_infos = self._analyze_movie()
     
     print("CURRICULUM_MOVIE: length=%s path=%s" % (self.movie_length, movie_path,), file=self.log_file)
     self.log_file.flush()

     self.episode = -1 
     self.total_steps = 0
     self.total_replay_steps = 0

   def _replay_reset(self):
     movie = retro.Movie(self.movie_path)
     movie.step()
     #self.unwrapped.em.set_state(movie.get_state())
     self.unwrapped.initial_state = movie.get_state()
     obs = super().reset()
     return movie, obs

   def _replay_step(self, movie):
     action_keys = []
     for i in range(self.unwrapped.NUM_BUTTONS):
       action_keys.append(movie.get_key(i))
     obs, reward, done, info = self.unwrapped.step(action_keys)
     return obs, reward, done, info, action_keys 

   def _analyze_movie(self):
     movie = self._replay_reset()[0]
     movie_length = 1
     movie_states = [ movie.get_state() ]
     movie_rewards = [ 0.0 ]
     movie_infos = [ {} ]
     done = False
     total_reward = 0
     while movie.step():
       _, reward, done, info, _ = self._replay_step(movie)
       total_reward += reward
       if done or info['screen_x'] == info['screen_x_end']:
         break
       movie_length += 1
       movie_states.append(self.unwrapped.em.get_state())
       movie_rewards.append(total_reward)
       movie_infos.append(info)
     return movie_length, movie_states, movie_rewards, movie_infos

   def reset(self):
     self.episode += 1
     success = False
     attempt = 0
     while not success:
       attempt += 1

       curriculum_progress = min(1.0,self.total_steps / max(1.0,float(self.max_curriculum_steps)))
       if self.randomize_curriculum:
         replay_length = max(0,min(self.movie_length-1, math.floor(random.random() * (1-curriculum_progress) * self.movie_length)))
       else:
         replay_length = max(0,min(self.movie_length-1, math.floor((1-curriculum_progress) * self.movie_length)))
         replay_length = math.floor(replay_length * 0.99 ** attempt)

       print("CURRICULUM_REPLAY_BEGIN: total_steps=%s total_replay_steps=%s episode=%s attempt=%s curriculum_progress=%s movie_length=%s replay_length=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, curriculum_progress, self.movie_length, replay_length), file=self.log_file)

       #done = False
       #replay_steps = 0
       #replay_reward = 0
       #movie, obs = self._replay_reset()
       #while not done and replay_steps < replay_length and movie.step():
       #  self.total_replay_steps += 1
       #  replay_steps += 1
       #  obs, reward, done, info, action_keys = self._replay_step(movie)
       #  print("CURRICULUM_STEP: total_steps=%s total_replay_steps=%s episode=%s attempt=%s movie_length=%s replay_length=%s replay_steps=%s action_keys=%s reward=%s replay_reward=%s done=%s info=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, self.movie_length, replay_length, replay_steps, action_keys, reward, replay_reward, done, info), file=self.log_file)
       #  replay_reward += reward
       
       #if not done:
       #  self.replay_reward = replay_reward
       #  success = True

       #self.unwrapped.em.set_state(self.movie_states[replay_length])
       self.unwrapped.initial_state = self.movie_states[replay_length]
       obs = super().reset()
       self.replay_reward = self.movie_rewards[replay_length]
       success = True
 
     print("CURRICULUM_REPLAY_END: total_steps=%s total_replay_steps=%s episode=%s attempt=%s movie_length=%s replay_length=%s replay_reward=%s" % (self.total_steps, self.total_replay_steps, self.episode, attempt, self.movie_length, replay_length, self.replay_reward), file=self.log_file)
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
