import numpy as np
import random

import tensorflow as tf

from gym import spaces
import ppo2.policies as policies

class PolicyExpert:
  def __init__(self, session, batch_size, policy_dir):
    self.session = session
    self.batch_size = batch_size
    self.policy = policies.CnnPolicy(self.session, np.zeros([84,84,4]), spaces.Discrete(7), batch_size, 4, reuse=False)
    self.action_probs = tf.nn.softmax(self.policy.pd.logits)
    self.policy_dir = policy_dir
  
  def initialize(self):
     saver = tf.train.Saver(var_list=tf.trainable_variables('ppo2_model'))
     latest_checkpoint = tf.train.latest_checkpoint(self.policy_dir)
     print("LOAD_POLICY_CHECKPOINT: %s" % (latest_checkpoint,))
     saver.restore(self.session, latest_checkpoint)

  def reset(self, num_actions, batch_size):
    assert num_actions == 7
    assert batch_size == self.batch_size

  def step(self, observations):
    return self.session.run(self.action_probs, feed_dict={self.policy.X:observations}) 
