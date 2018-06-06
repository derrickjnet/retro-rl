import re
import sys
import os
import csv
import cloudpickle
import gzip 
import itertools
import numpy as np
import tensorflow as tf
from sonic_util import SonicDiscretizer
import retro
import random
import cv2
cv2.ocl.setUseOpenCL(False)

def filter_action(a):
    a = a.copy()
    a[4] = False
    a[9] = False
    a[10] = False
    a[11] = False
    if a[6] == True and a[7] == True:
        a[6] = False
        a[7] = False
    if a[1] == True:
        a[0] = True
        a[1] = False
    if a[8] == True:
        a[0] = True
        a[8] = False
    if a[0] == True and (a[6] == True or a[7] == True):
        a[5] = False
    if a[0] == True and (a[6] or a[7]):
        next_a = a.copy()
        next_a[0] = False
        a[6] = False
        a[7] = False 
        return [a] + filter_action(next_a)
    return [a]

def warp_frame(frame, width=84, height=84):
   frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
   return frame[:, :, None]

def lookup_action(actions, a):
    if not np.any(a): return -1
    return [action_idx for (action_idx, action) in zip(range(len(actions)),actions) if np.array_equal(action, a)][0]

def replay_policy_targets(movie_path, game_act_name, event_file_name, obs_steps=4):
    print("ANALYZING: %s" % (movie_path,))
    event_writer = tf.python_io.TFRecordWriter(event_file_name + ".temp")
    game_name = '-'.join(game_act_name.split('-')[0:-1])
    act_name = game_act_name.split('-')[-1]
    obs = []

    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)

    env.initial_state = movie.get_state()
    env.reset()

    action_discretizer = SonicDiscretizer(env)

    total_steps = 0
    episode=0
    episode_step = 0
    while movie.step():
      total_steps += 1 

      action_keys = []
      for i in range(env.NUM_BUTTONS):
        action_keys.append(movie.get_key(i))
      new_obs, reward, done, info = env.step(action_keys)
      #new_obs, reward, done, info = env.step(filter_action(action_keys)[0])
      #new_obs, reward, done, info = env.step(random.choice(filter_action(action_keys)))
      total_reward = info['screen_x'] / max(1,float(info['screen_x_end'])) * 9000.0

      if total_steps % 4 != 1:
        continue

      episode_step += 1

      obs.append(warp_frame(new_obs))
      if len(obs) > obs_steps:
        obs = obs[-obs_steps:]
      elif len(obs) < obs_steps:
        obs_zeros = np.zeros_like(obs[-1], dtype=np.uint8)
        obs = list(itertools.repeat(obs_zeros, obs_steps - len(obs))) + obs 

      assert len(obs) == obs_steps
      obs_arr = np.dstack(obs)

      #policy_actions = list(map(lambda a: lookup_action(action_discretizer._actions, np.asarray(a)), filter_action(action_keys)))
      policy_actions = [-1]
      for policy_action in policy_actions: 
        policy_action_probs = np.zeros(len(action_discretizer._actions))
        if policy_action != -1:
          policy_action_probs[policy_action] = 1.0
        else:
          policy_action_probs[:] = 1.0 / len(policy_action_probs)
        print("STEP: game_act=%s total_steps=%s episode_step=%s total_reward=%s action_keys=%s action_probs=%s action=%s reward=%s done=%s info=%s" % (game_act_name, total_steps, episode_step, total_reward, action_keys, policy_action_probs, policy_action, reward, done, info))
        features = {
          'game_name' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(game_name)])),
          'act_name' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(act_name)])),
          'episode' : tf.train.Feature(int64_list=tf.train.Int64List(value=[episode])),
          'episode_step' : tf.train.Feature(int64_list=tf.train.Int64List(value=[episode_step])),
          'obs' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(obs_arr.tostring())])),
          'obs_shape' : tf.train.Feature(int64_list=tf.train.Int64List(value=obs_arr.shape)),
          'action_probs' : tf.train.Feature(float_list=tf.train.FloatList(value=policy_action_probs)),
          'action' : tf.train.Feature(int64_list=tf.train.Int64List(value=[policy_action])) 
        }
        record = tf.train.Example(features=tf.train.Features(feature=features))
        event_writer.write(record.SerializeToString())
    event_writer.close()
    os.rename(event_file_name + ".temp", event_file_name)

movie_path, event_path = (sys.argv[1], sys.argv[2])

game_act_name = movie_path.split("/")[-1]

event_file_name = event_path + "/" + game_act_name + ".events.tfrecords"
if not os.path.exists(event_file_name):
  replay_policy_targets(movie_path, game_act_name, event_file_name)
else:
  print("SKIPPING: game_act_name=%s" % (game_act_name,))
