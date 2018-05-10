#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import os
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import ppo2.ppo2 as ppo2
#import baselines.ppo2.ppo2 as ppo2
import ppo2.policies as policies
#import baselines.ppo2.policies as policies
import baselines.logger as logger
import gym_remote.exceptions as gre

from exploration.exploration_env import ExplorationEnv
from sonic_util import RewardScaler,AllowBacktracking,make_env

def main():
    discount = os.environ.get('RETRO_DISCOUNT')
    if discount != None:
      discount = float(discount)
    else:
      discount=0.99
    print("DISCOUNT: %s" % (discount,))

    """Run PPO until the environment throws an exception."""
    logger.configure(dir=os.environ.get('RETRO_LOGDIR'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=DummyVecEnv([lambda: RewardScaler(ExplorationEnv(AllowBacktracking(make_env())))]),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=discount, #0.99
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(1.5e6),
                   save_interval=1)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
