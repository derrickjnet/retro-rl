"""
Distributional Q-learning models.
"""

from abc import abstractmethod
from functools import partial
from math import log

import numpy as np
import tensorflow as tf

from anyrl.models.base import TFQNetwork
from anyrl.models.dqn_scalar import noisy_net_dense
from anyrl.models.util import nature_cnn, simple_mlp, take_vector_elems

import sys
import random
import datetime

def rainbow_models(session,
                   num_actions,
                   obs_vectorizer,
                   num_atoms=51,
                   min_val=-10,
                   max_val=10,
                   sigma0=0.5,
                   discount=0.99,
                   discover_steps=100000, 
                   cooling_steps=100000,
                   start_temperature=10.0,
                   stop_temperature=0.01,
                   expert_prob=0.01,
                   expert=None):
    """
    Create the models used for Rainbow
    (https://arxiv.org/abs/1710.02298).
    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.
    Returns:
      A tuple (online, target).
    """
    def maker(name):
        return NatureDistQNetwork(session, num_actions, obs_vectorizer, name,
                                  num_atoms, min_val, max_val, dueling=True,
                                  dense=partial(noisy_net_dense, sigma0=sigma0),
                                  discover_steps=discover_steps, cooling_steps=cooling_steps, start_temperature=start_temperature, stop_temperature=stop_temperature,
                                  expert_prob=expert_prob if name == 'dqn_online' else None,
                                  expert=expert if name == 'dqn_online' else None)
    return maker('dqn_online'), maker('dqn_target'), discount


class DistQNetwork(TFQNetwork):
    """
    An abstract Q-network that predicts action-conditional
    reward distributions (as opposed to expectations).
    Subclasses should override the base() and value_func()
    methods with specific neural network architectures.
    """

    def __init__(self, session, num_actions, obs_vectorizer, name, num_atoms, min_val, max_val,
                 dueling=False, dense=tf.layers.dense,
                 discover_steps=None, cooling_steps=None, start_temperature=None, stop_temperature=None, expert_prob=None, expert=None):
        """
        Create a distributional network.
        Args:
          session: the TF session.
          num_actions: size of action space.
          obs_vectorizer: observation vectorizer.
          name: name for this model.
          num_atoms: number of distribution atoms.
          min_val: minimum atom value.
          max_val: maximum atom value.
          dueling: if True, use a separate baseline and
            per-action value function.
          dense: the dense layer for use throughout the
            network.
        """
        super(DistQNetwork, self).__init__(session, num_actions, obs_vectorizer, name)
        self.dueling = dueling
        self.dense = dense
        self.discover_steps = discover_steps
        self.cooling_steps = cooling_steps
        self.expert_prob = expert_prob
        self.expert = expert
        self.dist = ActionDist(num_atoms, min_val, max_val)
        old_vars = tf.trainable_variables()
        with tf.variable_scope(name):
            #BEGIN: discover
            self.total_steps_var = tf.Variable(name="total_steps", dtype=tf.int64, initial_value=tf.constant(0,dtype=tf.int64), trainable=False) 
            self.total_steps_incr_op = tf.assign_add(self.total_steps_var, 1)
            #END: discover
            #BEGIN: soft q learning
            self.temperature = tf.cond(
                                self.total_steps_var <= discover_steps,
                                lambda: tf.maximum(1.0, start_temperature*(1.0-tf.cast(self.total_steps_var, tf.float32) / tf.cast(discover_steps, tf.float32))),
                                lambda: tf.maximum(stop_temperature, 1*(1.0-tf.cast(self.total_steps_var - discover_steps,tf.float32) / tf.cast(cooling_steps, tf.float32)))
                              )
            #END: soft q learning
            self.step_obs_ph = tf.placeholder(self.input_dtype,
                                              shape=(None,) + obs_vectorizer.out_shape)
            self.step_base_out = self.base(self.step_obs_ph)
            log_probs = self.value_func(self.step_base_out)
            values = self.dist.mean(log_probs)
            self.step_outs = (values, log_probs)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]

    @property
    def stateful(self):
        #BEGIN: discover
        #return False
        return True
        #END: discover

    def start_state(self, batch_size):
        #BEGIN: discover
        #return None
        self.expert.reset(self.num_actions, batch_size)
        if not hasattr(self, 'episode_idx'):
          self.episode_idx = 0
        else:
          self.episode_idx += 1
        return ([0 for _ in range(0, batch_size)], [False for _ in range(0, batch_size)])
        #END: discover

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values, dists = self.session.run(self.step_outs, feed_dict=feed)
        #BEGIN: discover
        expert_action_probs = self.expert.step(observations)
        total_steps = self.session.run(self.total_steps_incr_op)
        #END: discover
        #BEGIN: soft q learning
        actions = []
        temperature = self.session.run(self.temperature)
        for env_idx in range(0,len(observations)):
          episode_step = states[0][env_idx] + 1
          states[0][env_idx] = episode_step
          #BEGIN: discover
          if self.expert is not None:
            expert_flag = states[1][env_idx]
            if not expert_flag and random.random() > (1 - self.expert_prob) + self.expert_prob * min(1.0, float(total_steps) / self.discover_steps):
               expert_flag = True
            elif expert_flag and random.random() > 1 - self.expert_prob * min(1.0, float(total_steps) / self.discover_steps):
               expert_flag = False
            states[1][env_idx] = expert_flag
            if expert_flag:
              action_probs = expert_action_probs[env_idx]
              action_entropy = -sum([log(action_prob) * action_prob for action_prob in action_probs if action_prob > 0])
              action = np.random.choice(len(action_probs), p=action_probs) 
              print("EXPERT: timestamp=%s total_steps=%s env=%s episode=%s episode_step=%s action_probs=%s action_entropy=%s action=%s" % (datetime.datetime.now(), total_steps, env_idx, self.episode_idx, episode_step, list(action_probs), action_entropy, action))
              actions.append(action)
              continue 
          #END: discover
          action_values = values[env_idx, :]
          if temperature >= 0.001: 
            action_logits = (action_values - np.max(action_values))/temperature
            action_probs = np.exp(action_logits) / np.sum(np.exp(action_logits))
            action_entropy = -np.sum(action_probs * action_logits) + np.log(np.sum(np.exp(action_logits)))
            action = np.random.choice(len(action_probs), p=action_probs) 
            print("POLICY: timestamp=%s total_steps=%s env=%s episode=%s episode_step=%s temperature=%s action_values=%s action_probs=%s action_entropy=%s action=%s" % (datetime.datetime.now(), total_steps, env_idx, self.episode_idx, episode_step, temperature, list(action_values), list(action_probs), action_entropy, action))
          else:
            action = np.argmax(action_values)
            print("POLICY: timestamp=%s total_steps=%s env=%s episode=%s episode_step=%s action_values=%s action=%s" % (datetime.datetime.now(), total_steps, env_idx, self.episode_idx, episode_step, list(action_values), action))
          actions.append(action)
        sys.stdout.flush()
        #END: soft q learning
        return {
            #BEGIN: soft q learning
            #'actions': np.argmax(values, axis=1),
            'actions': actions,
            #END: soft q learning
            #BEGIN: discover 
            #'states': None,
            'states': states,
            #END: discover 
            'action_values': values,
            'action_dists': dists
        }

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        with tf.variable_scope(self.name, reuse=True):
            #BEGIN: soft q learning
            #max_actions = tf.argmax(self.dist.mean(self.value_func(self.base(new_obses))),
            #                        axis=1, output_type=tf.int32)
            action_values = self.dist.mean(self.value_func(self.base(new_obses)))
            action_probs = tf.cond(
                                self.temperature >= 0.001, 
                                lambda: tf.nn.softmax(action_values), 
                                lambda: tf.one_hot(tf.argmax(action_values, axis=1, output_type=tf.int32), self.num_actions)
                           )
            action_entropy = tf.cond(
                                  self.temperature >= 0.001, 
                                  lambda: tf.nn.softmax_cross_entropy_with_logits_v2(labels=action_probs, logits=action_values),
                                  lambda: 0.0
                                )
            #END: soft q learning
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals,
                                    tf.zeros_like(target_preds) - log(self.dist.num_atoms),
                                    target_preds)
        discounts = tf.where(terminals, tf.zeros_like(discounts), discounts)
        #BEGIN: soft q learning
        #target_dists = self.dist.add_rewards(tf.exp(take_vector_elems(target_preds, max_actions)),
        #                                     rews, discounts)
        target_dists = self.dist.add_rewards(tf.reduce_sum(tf.multiply(tf.exp(target_preds), tf.expand_dims(action_probs,-1)),1),
                                             rews + self.temperature*discounts*action_entropy, discounts)
        #END: soft q learning
        with tf.variable_scope(self.name, reuse=True):
            online_preds = self.value_func(self.base(obses))
            onlines = take_vector_elems(online_preds, actions)
            return _kl_divergence(tf.stop_gradient(target_dists), onlines)

    @property
    def input_dtype(self):
        return tf.float32

    @abstractmethod
    def base(self, obs_batch):
        """
        Go from a Tensor of observations to a Tensor of
        feature vectors to feed into the output heads.
        Returns:
          A Tensor of shape [batch_size x num_features].
        """
        pass

    def value_func(self, feature_batch):
        """
        Go from a 2-D Tensor of feature vectors to a 3-D
        Tensor of predicted action distributions.
        Args:
          feature_batch: a batch of features from base().
        Returns:
          A Tensor of shape [batch x actions x atoms].
        All probabilities are computed in the log domain.
        """
        logits = self.dense(feature_batch, self.num_actions * self.dist.num_atoms)
        actions = tf.reshape(logits, (tf.shape(logits)[0], self.num_actions, self.dist.num_atoms))
        if not self.dueling:
            return tf.nn.log_softmax(actions)
        values = tf.expand_dims(self.dense(feature_batch, self.dist.num_atoms), axis=1)
        actions -= tf.reduce_mean(actions, axis=1, keepdims=True)
        return tf.nn.log_softmax(values + actions)

    # pylint: disable=W0613
    def step_feed_dict(self, observations, states):
        """Produce a feed_dict for taking a step."""
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations)}


class MLPDistQNetwork(DistQNetwork):
    """
    A multi-layer perceptron distributional Q-network.
    This is the distributional equivalent of MLPQNetwork.
    """

    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 layer_sizes,
                 activation=tf.nn.relu,
                 dueling=False,
                 dense=tf.layers.dense):
        self.layer_sizes = layer_sizes
        self.activation = activation
        super(MLPDistQNetwork, self).__init__(session, num_actions, obs_vectorizer, name, num_atoms,
                                              min_val, max_val, dueling=dueling, dense=dense)

    def base(self, obs_batch):
        return simple_mlp(obs_batch, self.layer_sizes, self.activation, dense=self.dense)


class NatureDistQNetwork(DistQNetwork):
    """
    A distributional Q-network model based on the Nature
    DQN paper.
    This is the distributional equivalent of NatureQNetwork.
    """

    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 dueling=False,
                 dense=tf.layers.dense,
                 discover_steps=None, cooling_steps=None, start_temperature=None, stop_temperature=None, 
                 expert_prob=None, expert=None,
                 input_dtype=tf.uint8,
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super(NatureDistQNetwork, self).__init__(session, num_actions, obs_vectorizer, name,
                                                 num_atoms, min_val, max_val,
                                                 dueling=dueling, dense=dense,
                                                 discover_steps=discover_steps, cooling_steps=cooling_steps, start_temperature=start_temperature, stop_temperature=stop_temperature,
                                                 expert_prob=expert_prob, expert=expert)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        return nature_cnn(obs_batch, dense=self.dense)


class ActionDist:
    """
    A discrete reward distribution.
    """

    def __init__(self, num_atoms, min_val, max_val):
        assert num_atoms >= 2
        assert max_val > min_val
        self.num_atoms = num_atoms
        self.min_val = min_val
        self.max_val = max_val
        self._delta = (self.max_val - self.min_val) / (self.num_atoms - 1)

    def atom_values(self):
        """Get the reward values for each atom."""
        return [self.min_val + i * self._delta for i in range(0, self.num_atoms)]

    def mean(self, log_probs):
        """Get the mean rewards for the distributions."""
        probs = tf.exp(log_probs)
        return tf.reduce_sum(probs * tf.constant(self.atom_values(), dtype=probs.dtype), axis=-1)

    def add_rewards(self, probs, rewards, discounts):
        """
        Compute new distributions after adding rewards to
        old distributions.
        Args:
          log_probs: a batch of log probability vectors.
          rewards: a batch of rewards.
          discounts: the discount factors to apply to the
            distribution rewards.
        Returns:
          A new batch of log probability vectors.
        """
        atom_rews = tf.tile(tf.constant([self.atom_values()], dtype=probs.dtype),
                            tf.stack([tf.shape(rewards)[0], 1]))

        fuzzy_idxs = tf.expand_dims(rewards, axis=1) + tf.expand_dims(discounts, axis=1) * atom_rews
        fuzzy_idxs = (fuzzy_idxs - self.min_val) / self._delta

        # If the position were exactly 0, rounding up
        # and subtracting 1 would cause problems.
        fuzzy_idxs = tf.clip_by_value(fuzzy_idxs, 1e-18, float(self.num_atoms - 1))

        indices_1 = tf.cast(tf.ceil(fuzzy_idxs) - 1, tf.int32)
        fracs_1 = tf.abs(tf.ceil(fuzzy_idxs) - fuzzy_idxs)
        indices_2 = indices_1 + 1
        fracs_2 = 1 - fracs_1

        res = tf.zeros_like(probs)
        for indices, fracs in [(indices_1, fracs_1), (indices_2, fracs_2)]:
            index_matrix = tf.expand_dims(tf.range(tf.shape(indices)[0], dtype=tf.int32), axis=1)
            index_matrix = tf.tile(index_matrix, (1, self.num_atoms))
            scatter_indices = tf.stack([index_matrix, indices], axis=-1)
            res = res + tf.scatter_nd(scatter_indices, probs * fracs, tf.shape(res))

        return res


def _kl_divergence(probs, log_probs):
    masked_diff = tf.where(tf.equal(probs, 0), tf.zeros_like(probs), tf.log(probs) - log_probs)
    return tf.reduce_sum(probs * masked_diff, axis=-1)
