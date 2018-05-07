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
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from dqn.move_expert import MoveExpert
from dqn.dqn_scalar import noisy_net_models
from dqn.dqn_dist import rainbow_models
from sonic_util import AllowBacktracking,make_env
from exploration.exploration_env import ExplorationEnv

def main():
    """Run DQN until the environment throws an exception."""
    env = ExplorationEnv(AllowBacktracking(make_env(stack=False)))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    config.log_device_placement=True
    with tf.device(os.environ.get("RETRO_DEVICE", '/gpu:0')):
      with tf.Session(config=config) as sess:
        if os.environ['RETRO_DQN'] == 'noisy_net':
          dqn = DQN(*noisy_net_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  discount=0.999, #0.99
                                  expert = MoveExpert()
                                 ))
        elif os.environ['RETRO_DQN'] == 'rainbow':
          dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  num_atoms=101,
                                  min_val=-1000, #-200
                                  max_val=1000, #200
                                  discount=0.999, #0.99
                                  expert = MoveExpert()
                                 ))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=1500000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(1500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
