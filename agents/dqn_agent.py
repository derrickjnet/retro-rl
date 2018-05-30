#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf
import os

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from rollouts.batched_player import BatchedPlayer
from anyrl.rollouts import PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from dqn.random_move_expert import RandomMoveExpert
from dqn.policy_expert import PolicyExpert
from dqn.dqn_scalar import noisy_net_models as noisy_net_models
from dqn.dqn_dist import rainbow_models as rainbow_models
from dqn.soft_dqn_scalar import noisy_net_models as soft_noisy_net_models
from dqn.soft_dqn_dist import rainbow_models as soft_rainbow_models
from sonic_util import make_batched_env
from exploration.exploration import Exploration
from exploration.exploration_batched_env import ExplorationBatchedEnv
from exploration.state_encoder import StateEncoder

class ScheduledSaver:
  def __init__(self, sess, fname, save_steps=10000):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    self.sess = sess
    self.fname = fname
    self.scheduler_saver = tf.train.Saver(max_to_keep=None)
    self.save_steps = save_steps
    self.total_steps = 0
    self.last_save = 0

  def handle_episode(self, steps):
    self.total_steps += steps
    if self.last_save + self.save_steps < self.total_steps:
      print("DQN_SAVE_CHECKPOINT: total_steps=%s last_save=%s" % (self.total_steps, self.last_save))
      self.do_save()
      self.last_save = self.total_steps

  def do_save(self):
      self.scheduler_saver.save(self.sess, self.fname + "/checkpoint", global_step=self.total_steps)

def main():
    discount = os.environ.get('RETRO_DISCOUNT')
    if discount != None:
      discount = float(discount)
    else:
      discount=0.99
    print("DISCOUNT: %s" % (discount,))

    """Run DQN until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    config.log_device_placement=False
    with tf.Session(config=config) as sess:
      state_encoder = StateEncoder(sess)

      env = make_batched_env()
      env_ids = env.env_ids
      env = BatchedFrameStack(env, num_images=4, concat=True)
      env.env_ids = env_ids 
      env = ExplorationBatchedEnv(env, Exploration, state_encoder=state_encoder)

      if 'RETRO_POLICY_DIR' in os.environ:
        expert = PolicyExpert(sess, batch_size=1, policy_dir=os.environ['RETRO_POLICY_DIR'])
      else:
        expert = RandomMoveExpert()

      if os.environ['RETRO_DQN'] == 'noisy_net':
        dqn = DQN(*noisy_net_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  discount=discount, #0.99
                                  expert = expert
                                 ))
      elif os.environ['RETRO_DQN'] == 'rainbow':
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  num_atoms=101,
                                  min_val=-1000, #-200
                                  max_val=1000, #200
                                  discount=discount, #0.99
                                  expert = expert
                                 ))
      elif os.environ['RETRO_DQN'] == 'soft_noisy_net':
        dqn = DQN(*soft_noisy_net_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  discount=discount, #0.99
                                  expert = expert
                                 ))
      elif os.environ['RETRO_DQN'] == 'soft_rainbow':
          dqn = DQN(*soft_rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  num_atoms=101,
                                  min_val=-1000, #-200
                                  max_val=1000, #200
                                  discount=discount, #0.99
                                  expert = expert
                                 ))
      scheduler_saver = ScheduledSaver(sess, os.environ["RETRO_CHECKPOINTDIR"] + "/tensorflow/")
      player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
      optimize = dqn.optimize(learning_rate=1e-4)
      sess.run(tf.global_variables_initializer())
      if 'RETRO_INIT_DIR' in os.environ:
        saver = tf.train.Saver(var_list=list(filter(lambda v: not 'sigma' in v.name, tf.trainable_variables('^dqn_model/'))))
        latest_checkpoint = tf.train.latest_checkpoint(os.environ['RETRO_INIT_DIR'])
        print("DQN_INIT_CHECKPOINT: %s" % (latest_checkpoint,))
        saver.restore(sess, latest_checkpoint)
        from tensorflow.python.tools import inspect_checkpoint as chkp
        chkp.print_tensors_in_checkpoint_file(latest_checkpoint,'',all_tensors=True) 
      state_encoder.initialize()
      expert.initialize()
      dqn.train(num_steps=1000000, # Make sure an exception arrives before we stop.
                player=player,
                replay_buffer=PrioritizedReplayBuffer(250000, 0.5, 0.4, epsilon=0.1),
                optimize_op=optimize,
                train_interval=1,
                target_interval=8192,
                batch_size=32,
                min_buffer_size=20000,
                handle_ep = lambda steps,rew: scheduler_saver.handle_episode(steps))

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
