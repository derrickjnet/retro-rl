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
  def __init__(self, fname, save_steps=10000):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    self.fname = fname
    self.saver = tf.train.Saver(max_to_keep=None)
    self.save_steps = save_steps
    self.total_steps = 0
    self.last_save = 0

  def handle_episode(self, steps):
    self.total_steps += steps
    if self.last_save + self.save_steps < self.total_steps:
      print("CHECKPOINT: total_steps=%s last_save=%s" % (self.total_steps, self.last_save))
      self.do_save()
      self.last_save = self.total_steps

  def do_save(self):
      self.saver.save(tf.get_default_session(), self.fname + "/checkpoint", global_step=self.total_steps)

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
    config.log_device_placement=True
    with tf.Session(config=config) as sess:
      with tf.device(os.environ.get("RETRO_DEVICE", '/gpu:0')):
        if 'RETRO_ENCODER_DIR' in os.environ:
          state_encoder = StateEncoder(sess, encoder_dir = os.environ['RETRO_ENCODER_DIR'])
        else:
          state_encoder = None

        env = make_batched_env()
        env_ids = env.env_ids
        env = BatchedFrameStack(env, num_images=4, concat=True)
        env.env_ids = env_ids 
        env = ExplorationBatchedEnv(env, Exploration, state_encoder=state_encoder)

        if 'RETRO_POLICYDIR' in os.environ:
          expert = PolicyExpert(sess, batch_size=1, policy_dir=os.environ['RETRO_POLICYDIR'])
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
      saver = ScheduledSaver(os.environ["RETRO_CHECKPOINT_DIR"] + "/checkpoints/")
      player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
      optimize = dqn.optimize(learning_rate=1e-4)
      sess.run(tf.global_variables_initializer())
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
                handle_ep = lambda steps,rew: saver.handle_episode(steps))

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
