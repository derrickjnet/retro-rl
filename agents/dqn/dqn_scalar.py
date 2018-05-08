"""
Scalar Q-learning models.
"""

from abc import abstractmethod
from functools import partial
from math import sqrt,log

import numpy as np
import tensorflow as tf

from anyrl.models.base import TFQNetwork
from anyrl.models.util import nature_cnn, nature_huber_loss, simple_mlp, take_vector_elems

import sys
import random
import datetime

import sys
import random
import datetime

def noisy_net_models(session,
                   num_actions,
                   obs_vectorizer,
                   sigma0=0.5,
                   discount=0.99,
                   exploration_steps=100000, 
                   expert_prob=0.01,
                   expert=None):
    """
    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      sigma0: initial Noisy Net noise.
    Returns:
      A tuple (online, target).
    """
    def maker(name):
        return NatureQNetwork(session, num_actions, obs_vectorizer, name,
                                  dueling=True,
                                  dense=partial(noisy_net_dense, sigma0=sigma0),
                                  exploration_steps=exploration_steps,
                                  expert_prob=expert_prob, expert=expert if name == 'online' else None)
    return maker('online'), maker('target'), discount

class ScalarQNetwork(TFQNetwork):
    """
    An abstract Q-network that predicts action values as
    scalars (as opposed to distributions).
    Subclasses should override the base() and value_func()
    methods with specific neural network architectures.
    """

    def __init__(self, session, num_actions, obs_vectorizer, name,
                 dueling=False, dense=tf.layers.dense, loss_fn=tf.square,
                 exploration_steps=None, expert_prob=None, expert=None):
        super(ScalarQNetwork, self).__init__(session, num_actions, obs_vectorizer, name)
        self.dueling = dueling
        self.dense = dense
        self.loss_fn = loss_fn
        self.exploration_steps = exploration_steps
        self.expert_prob = expert_prob
        self.expert = expert
        old_vars = tf.trainable_variables()
        with tf.variable_scope(name):
            #BEGIN: exploration
            self.total_steps_var = tf.Variable(name="total_steps", dtype=tf.int64, initial_value=tf.constant(0,dtype=tf.int64), trainable=False) 
            self.total_steps_incr_op = tf.assign_add(self.total_steps_var, 1)    
            #END: exploration
            self.step_obs_ph = tf.placeholder(self.input_dtype,
                                              shape=(None,) + obs_vectorizer.out_shape)
            self.step_base_out = self.base(self.step_obs_ph)
            self.step_values = self.value_func(self.step_base_out)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]

    @property
    def stateful(self):
        #BEGIN: exploration
        #return False
        return True
        #END: exploration

    def start_state(self, batch_size):
        #BEGIN: exploration
        #return None
        self.expert.reset(self.num_actions, batch_size)
        if not hasattr(self, 'episode_idx'):
          self.episode_idx = 0
        else:
          self.episode_idx += 1
        return ([0 for _ in range(0, batch_size)], [False for _ in range(0, batch_size)])
        #END: exploration

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values = self.session.run(self.step_values, feed_dict=feed)
        #BEGIN: exploration
        expert_action_probs = self.expert.step(observations)
        total_steps = self.session.run(self.total_steps_incr_op)
        #END: exploration
        actions = []
        for env_idx in range(0,len(observations)):
          episode_step = states[0][env_idx] + 1
          states[0][env_idx] = episode_step
          #BEGIN: exploration
          if self.expert is not None:
            expert_flag = states[1][env_idx]
            if not expert_flag and random.random() > (1 - self.expert_prob) + self.expert_prob * min(1.0, float(total_steps) / self.exploration_steps):
               expert_flag = True
            elif expert_flag and random.random() > 1 - self.expert_prob * min(1.0, float(total_steps) / self.exploration_steps):
               expert_flag = False
            states[1][env_idx] = expert_flag
            if expert_flag:
              action_probs = expert_action_probs[env_idx]
              action_entropy = -sum([log(action_prob) * action_prob for action_prob in action_probs if action_prob > 0])
              action = np.random.choice(len(action_probs), p=action_probs) 
              print("EXPERT: timestamp=%s total_steps=%s env=%s episode=%s episode_step=%s action_probs=%s action_entropy=%s action=%s" % (datetime.datetime.now(), total_steps, env_idx, self.episode_idx, episode_step, list(action_probs), action_entropy, action))
              actions.append(action)
              continue 
          #END: exploration 
          action_values = values[env_idx, :]
          action = np.argmax(action_values)
          print("POLICY: timestamp=%s total_steps=%s env=%s episode=%s episode_step=%s action_values=%s action=%s" % (datetime.datetime.now(), total_steps, env_idx, self.episode_idx, episode_step, list(action_values), action))
          actions.append(action)
        sys.stdout.flush()
        return {
            #BEGIN: exploration 
            #'actions': np.argmax(values, axis=1),
            #'states': None,
            'actions': actions,
            'states': states,
            #END: exploration 
            'action_values': values
        }

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        with tf.variable_scope(self.name, reuse=True):
            max_actions = tf.argmax(self.value_func(self.base(new_obses)),
                                    axis=1, output_type=tf.int32)
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals, tf.zeros_like(target_preds), target_preds)
        targets = rews + discounts * take_vector_elems(target_preds, max_actions)
        with tf.variable_scope(self.name, reuse=True):
            online_preds = self.value_func(self.base(obses))
            onlines = take_vector_elems(online_preds, actions)
            return self.loss_fn(onlines - tf.stop_gradient(targets))

    @property
    def input_dtype(self):
        return tf.float32

    @abstractmethod
    def base(self, obs_batch):
        """
        Go from a Tensor of observations to a Tensor of
        feature vectors to feed into the final layer.
        Returns:
          A Tensor of shape [batch_size x num_features].
        """
        pass

    def value_func(self, feature_batch):
        """
        Go from a Tensor of feature vectors to a Tensor of
        predicted action values.
        Args:
          feature_batch: a batch of features from base().
        Returns:
          A Tensor of shape [batch_size x num_actions].
        """
        if not self.dueling:
            return self.dense(feature_batch, self.num_actions)
        values = self.dense(feature_batch, 1)
        actions = self.dense(feature_batch, self.num_actions)
        actions -= tf.reduce_mean(actions, axis=1, keepdims=True)
        return values + actions

    # pylint: disable=W0613
    def step_feed_dict(self, observations, states):
        """Produce a feed_dict for taking a step."""
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations)}


class MLPQNetwork(ScalarQNetwork):
    """
    A multi-layer perceptron Q-network.
    """

    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 layer_sizes,
                 activation=tf.nn.relu,
                 dueling=False,
                 dense=tf.layers.dense,
                 loss_fn=tf.square):
        """
        Create an MLP model.
        Args:
          session: the TF session used by step().
          num_actions: the number of possible actions.
          obs_vectorizer: a vectorizer for the observation
            space.
          name: the scope name for the model. This should
            be different for the target and online models.
          layer_sizes: sequence of hidden layer sizes.
          activation: the activation function.
          dueling: use a dueling architecture.
          dense: the dense layer function.
          loss_fn: the target loss function.
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        super(MLPQNetwork, self).__init__(session, num_actions, obs_vectorizer, name,
                                          dueling=dueling, dense=dense, loss_fn=loss_fn)

    def base(self, obs_batch):
        return simple_mlp(obs_batch, self.layer_sizes, self.activation, dense=self.dense)


class NatureQNetwork(ScalarQNetwork):
    """
    A Q-network model based on the Nature DQN paper.
    """

    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 dueling=False,
                 dense=tf.layers.dense,
                 loss_fn=nature_huber_loss,
                 exploration_steps=None, 
                 expert_prob=None, expert=None,
                 input_dtype=tf.uint8,
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super(NatureQNetwork, self).__init__(session, num_actions, obs_vectorizer, name,
                                             dueling=dueling, dense=dense, loss_fn=loss_fn,
                                             exploration_steps=exploration_steps,
                                             expert_prob=expert_prob, expert=expert)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        return nature_cnn(obs_batch, dense=self.dense)


class EpsGreedyQNetwork(TFQNetwork):
    """
    A wrapper around a Q-network that adds epsilon-greedy
    exploration to the actions.
    The epsilon parameter can be any object that supports
    float() conversion, including TFScheduleValue.
    """

    def __init__(self, model, epsilon):
        super(EpsGreedyQNetwork, self).__init__(model.session, model.num_actions,
                                                model.obs_vectorizer, model.name)
        self.model = model
        self.epsilon = epsilon

    @property
    def stateful(self):
        return self.model.stateful

    def start_state(self, batch_size):
        return self.model.start_state(batch_size)

    def step(self, observations, states):
        result = self.model.step(observations, states)
        new_actions = []
        eps = float(self.epsilon)
        for action in result['actions']:
            if random.random() < eps:
                new_actions.append(random.randrange(self.num_actions))
            else:
                new_actions.append(action)
        result['actions'] = new_actions
        return result

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        return self.model.transition_loss(target_net.model, obses, actions, rews, new_obses,
                                          terminals, discounts)

    @property
    def input_dtype(self):
        return self.model.input_dtype


def noisy_net_dense(inputs,
                    units,
                    activation=None,
                    sigma0=0.5,
                    kernel_initializer=None,
                    name=None,
                    reuse=None):
    """
    Apply a factorized Noisy Net layer.
    See https://arxiv.org/abs/1706.10295.
    Args:
      inputs: the batch of input vectors.
      units: the number of output units.
      activation: the activation function.
      sigma0: initial stddev for the weight noise.
      kernel_initializer: initializer for kernels. Default
        is to use Gaussian noise that preserves stddev.
      name: the name for the layer.
      reuse: reuse the variable scope.
    """
    num_inputs = inputs.get_shape()[-1].value
    stddev = 1 / sqrt(num_inputs)
    activation = activation if activation is not None else (lambda x: x)
    if kernel_initializer is None:
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    with tf.variable_scope(None, default_name=(name or 'noisy_layer'), reuse=reuse):
        weight_mean = tf.get_variable('weight_mu',
                                      shape=(num_inputs, units),
                                      initializer=kernel_initializer)
        bias_mean = tf.get_variable('bias_mu',
                                    shape=(units,),
                                    initializer=tf.zeros_initializer())
        stddev *= sigma0
        weight_stddev = tf.get_variable('weight_sigma',
                                        shape=(num_inputs, units),
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_stddev = tf.get_variable('bias_sigma',
                                      shape=(units,),
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_noise = tf.random_normal((units,), dtype=bias_stddev.dtype.base_dtype)
        weight_noise = _factorized_noise(num_inputs, units)
        return activation(tf.matmul(inputs, weight_mean + weight_stddev * weight_noise) +
                          bias_mean + bias_stddev * bias_noise)


def _factorized_noise(inputs, outputs):
    noise1 = _signed_sqrt(tf.random_normal((inputs, 1)))
    noise2 = _signed_sqrt(tf.random_normal((1, outputs)))
    return tf.matmul(noise1, noise2)


def _signed_sqrt(values):
    return tf.sqrt(tf.abs(values)) * tf.sign(values)