import re
import sys
import os
import csv
import cloudpickle
import gzip 
import itertools
import numpy as np

def collect_policy_targets(root_path, game_act_name, start_step, stop_step, target_key, event_file_name, obs_steps=4):
    event_file = gzip.open(event_file_name + ".temp", "wb")
    game_name = '-'.join(game_act_name.split('-')[0:-1])
    act_name = game_act_name.split('-')[-1]
    obs = []
    obs_total_steps = None
    obs_episode = None
    obs_episode_step = None
    policy_total_steps = None
    policy_episode = None
    policy_episode_step = None
    policy_targets = None
    policy_action = None
    for line in open(root_path + '/' + game_act_name + '/log', 'r'):
      key_values = dict(re.findall(r'(\S+)=(\[[^\]]*\]|{[^}]*}|[^ ]*)', line.rstrip()))
      if not 'total_steps' in key_values: continue
      total_steps = int(key_values['total_steps'])
      print("PROCESSING: game=%s act=%s total_steps=%s" % (game_name, act_name, total_steps,))
      if total_steps < start_step: continue
      if total_steps > stop_step: break
      if line.startswith("POLICY"):
        policy_total_steps = total_steps
        policy_action = int(key_values['action'])
        policy_targets = eval(key_values[target_key])
      elif line.startswith("STEP:"):
        episode = int(key_values['episode'])
        episode_step = int(key_values['episode_step'])
        action = int(key_values['action'])
        if obs_total_steps == total_steps-1 and obs_episode == episode and policy_total_steps == total_steps:
          assert obs_episode_step == episode_step - 1
          assert policy_action == action
          obs_arr = np.dstack(obs)
          print("EVENT: game=%s act=%s total_steps=%s episode=%s episode_step=%s obs=%s targets=%s action=%s" % (game_name, act_name, total_steps, episode, episode_step, obs_arr.shape, policy_targets, policy_action))
          assert len(obs) == obs_steps
          event = { 'game_name' : game_name, 'act_name' : act_name, 'total_steps' : total_steps, 'episode' : episode, 'episode_step' : episode_step, 'obs' : obs_arr, 'targets' : policy_targets, 'action' : policy_action }
          cloudpickle.dump(event, event_file)
        step_data = cloudpickle.load(gzip.open(root_path + '/' + game_act_name + '/' + game_act_name + "-" + str(episode).zfill(4) + ".steps/" + str(episode) + "." + str(episode_step) + ".step.gz", 'rb'))
        assert int(step_data['total_steps']) == total_steps
        assert int(step_data['episode']) == episode
        assert int(step_data['episode_step']) == episode_step
        assert int(step_data['action']) == action
        obs_total_steps = total_steps
        obs_episode = episode
        obs_episode_step = episode_step
        obs_new = step_data['obs']
        if not hasattr(obs_new, 'shape'):
          obs_new = obs_new.__array__()
        if len(obs_new.shape) == 3:
          obs = [obs_new[:,:,i] for i in range(0,obs_new.shape[2])] 
        else:
          obs.append(obs_new)
        if len(obs) > obs_steps:
          obs = obs[-obs_steps:]
        elif len(obs) < obs_steps:
          obs_zeros = np.zeros_like(obs[-1], dtype=float)
          obs = list(itertools.repeat(obs_zeros, obs_steps - len(obs))) + obs 
    event_file.close()
    os.rename(event_file_name + ".temp", event_file_name)

root_path, game_act_name, start_step, stop_step, target_key, event_path = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])

event_file_name = event_path + "/" + game_act_name + ".events.gz"
if not os.path.exists(event_file_name):
  collect_policy_targets(root_path, game_act_name, start_step, stop_step, target_key, event_file_name)
else:
  print("SKIPPING: game_act_name=%s" % (game_act_name,))
