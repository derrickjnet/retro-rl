import os
import datetime
import numpy as np
import random
import tensorflow as tf
from exploration.autoencoder import Autoencoder 

class StateEncoder:
  def __init__(self, sess, num_actions, num_images, embedder_dir=os.environ.get('RETRO_STATE_EMBEDDER_DIR'), embedder_model_scope=os.environ.get('RETRO_STATE_EMBEDDER_MODEL_SCOPE', 'state_embedder'), embedder_nfilters=int(os.environ.get('RETRO_STATE_EMBEDDER_NFILTERS',32)), embedder_embedding_size=int(os.environ.get('RETRO_STATE_EMBEDDER_EMBEDDING_SIZE',256)), predictor_dir=os.environ.get('RETRO_STATE_PREDICTOR_DIR'), predictor_model_scope=os.environ.get('RETRO_STATE_PREDICTOR_MODEL_SCOPE', 'state_predictor'), predictor_nfilters=int(os.environ.get('RETRO_STATE_PREDICTOR_NFILTES', 32)), predictor_embedding_size=int(os.environ.get('RETRO_STATE_PREDICTOR_EMBEDDING_SIZE', 256)), predictor_replay_size=int(os.environ.get("RETRO_STATE_PREDICTOR_REPLAY_SIZE", 20000)), predictor_batch_size=int(os.environ.get("RETRO_STATE_PREDICTOR_BATCH_SIZE", 64))):
    self.sess = sess
    self.num_actions = num_actions
    self.num_images = num_images
    self.embedder_dir = embedder_dir
    self.predictor_dir = predictor_dir
    if self.embedder_dir:
      self.embedder = Autoencoder(nfilters=embedder_nfilters, embedding_size=embedder_embedding_size)
      self.embedder_model_scope = embedder_model_scope
      with tf.variable_scope(embedder_model_scope):
        self.embedder_model_obses, _, self.embedder_model_embeddings = self.embedder.embed(embedding_activation=tf.nn.sigmoid) 
  
    if self.predictor_dir:
      self.predictor = Autoencoder(nfilters=predictor_nfilters, embedding_size=predictor_embedding_size)
      self.predictor_model_scope = predictor_model_scope

      self.predictor_eval_step = 0
      
      predictor_model_builder = lambda predictor_rescaled_obses, predictor_actions: self.build_inverse_forward_predictor_model(predictor_rescaled_obses, predictor_actions)
    
      self.eval_predictor_model_obses, self.eval_predictor_actions, self.eval_predictor_model_rescaled_obses = self.build_predictor_placeholders()
      self.eval_predictor_extras, self.eval_predictor_rewards, _ = predictor_model_builder(self.eval_predictor_model_rescaled_obses, self.eval_predictor_actions)
    
      self.train_predictor_model_obses, self.train_predictor_actions, self.train_predictor_model_rescaled_obses = self.build_predictor_placeholders()
      self.train_predictor_extras, self.train_predictor_rewards, self.predictor_train_loss = predictor_model_builder(self.train_predictor_model_rescaled_obses, self.train_predictor_actions)

      with tf.variable_scope(predictor_model_scope + "_train"):
        self.predictor_batch_size = predictor_batch_size 
        self.predictor_train_global_step = tf.train.create_global_step()
        self.predictor_train_step = tf.train.AdamOptimizer(learning_rate=0.0001*self.predictor_batch_size).minimize(self.predictor_train_loss, global_step = self.predictor_train_global_step)
        self.predictor_replay_buffer = []
        self.predictor_replay_size = predictor_replay_size

  def build_predictor_placeholders(self):
    with tf.variable_scope(self.predictor_model_scope, reuse=tf.AUTO_REUSE):
      predictor_actions = tf.placeholder(tf.int32, [None])
      with tf.variable_scope("observations"):
        predictor_model_obses = tf.placeholder(tf.uint8, [None, 84, 84, self.num_images])
      predictor_model_rescaled_obses = self.predictor.observations_rescaled(predictor_model_obses)
    return predictor_model_obses, predictor_actions, predictor_model_rescaled_obses

  def build_autoencoder_predictor_model(self, predictor_model_rescaled_obses, predictor_actions):
    with tf.variable_scope(self.predictor_model_scope, reuse=tf.AUTO_REUSE):
      predictor_model_rescaled_obses = predictor_model_rescaled_obses[:,:,:,-1:]
      predictor_model_embeddings = self.predictor.encoder(predictor_model_rescaled_obses, embedding_activation=None, reuse=tf.AUTO_REUSE)
      predictor_model_outputs = self.predictor.decoder(predictor_model_embeddings)
      predictor_reconstruction_errors = self.predictor.reconstruction_errors(predictor_model_rescaled_obses, predictor_model_outputs)
      predictor_reconstruction_loss = tf.reduce_mean(predictor_reconstruction_errors)
      predictor_embedding_loss = self.predictor.embedding_loss(predictor_model_embeddings)
    with tf.variable_scope(self.predictor_model_scope + "_internal", reuse=tf.AUTO_REUSE):
      predictor_extras = tf.zeros(0)
      predictor_rewards = 10 * tf.sqrt(predictor_reconstruction_errors / 84 / 84)
      predictor_train_loss = predictor_reconstruction_loss + predictor_embedding_loss
    return predictor_extras, predictor_rewards, predictor_train_loss

  def build_inverse_forward_predictor_model(self, predictor_model_rescaled_obses, predictor_actions, beta=0.2):
    inverse_predictor_extras, inverse_predictor_rewards, inverse_predictor_train_loss = self.build_inverse_predictor_model(predictor_model_rescaled_obses, predictor_actions)
    forward_predictor_extras, forward_predictor_rewards, forward_predictor_train_loss = self.build_forward_predictor_model(predictor_model_rescaled_obses, predictor_actions)

    with tf.variable_scope(self.predictor_model_scope + "_internal", reuse=tf.AUTO_REUSE):
      predictor_extras = tf.stack([inverse_predictor_train_loss, forward_predictor_train_loss])
      predictor_rewards = forward_predictor_rewards 
      predictor_train_loss = (1-beta) * inverse_predictor_train_loss + beta * forward_predictor_train_loss
 
    return predictor_extras, predictor_rewards, predictor_train_loss

  def build_forward_predictor_model(self, predictor_model_rescaled_obses, predictor_actions):
    with tf.variable_scope(self.predictor_model_scope, reuse=tf.AUTO_REUSE):
      predictor_model_embeddings = [self.predictor.encoder(predictor_model_rescaled_obses[:,:,:,i:i+1], embedding_activation=None, reuse=tf.AUTO_REUSE) for i in range(self.num_images)]
    with tf.variable_scope(self.predictor_model_scope + "_internal", reuse=tf.AUTO_REUSE):
      predictor_net = tf.layers.dense(tf.concat(predictor_model_embeddings[:-1] + [tf.one_hot(predictor_actions, self.num_actions)], -1), 256, activation=tf.nn.relu, name="forward_fc1")
      predictor_embedding = tf.layers.dense(predictor_net, self.predictor.embedding_size, name="forward_fc2")
      predictor_forward_errors = 0.5*tf.reduce_sum(tf.square(tf.subtract(predictor_embedding, tf.stop_gradient(predictor_model_embeddings[-1]))), -1)
      predictor_extras = tf.expand_dims(tf.sqrt(tf.reduce_mean(tf.square(predictor_model_embeddings[-1]), -1)),-1)
      predictor_rewards = predictor_forward_errors
      predictor_train_loss = tf.reduce_mean(predictor_forward_errors)
    return predictor_extras, predictor_rewards, predictor_train_loss

  def build_inverse_predictor_model(self, predictor_model_rescaled_obses, predictor_actions):
    with tf.variable_scope(self.predictor_model_scope, reuse=tf.AUTO_REUSE):
      predictor_model_embeddings = [self.predictor.encoder(predictor_model_rescaled_obses[:,:,:,i:i+1], embedding_activation=None, reuse=tf.AUTO_REUSE) for i in range(self.num_images)]
    with tf.variable_scope(self.predictor_model_scope + "_internal", reuse=tf.AUTO_REUSE):
      predictor_net = tf.layers.dense(tf.concat(predictor_model_embeddings, -1), 256, activation=tf.nn.relu, name="inverse_fc1")
      predictor_logits = tf.layers.dense(predictor_net, self.num_actions, name="inverse_fc2")
      predictor_inverse_errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictor_logits, labels=predictor_actions)
      predictor_extras = tf.nn.softmax(predictor_logits,-1)
      predictor_rewards = predictor_inverse_errors
      predictor_train_loss = tf.reduce_mean(predictor_inverse_errors)
    return predictor_extras, predictor_rewards, predictor_train_loss
  
  def initialize(self):
    if self.embedder_dir:
      embedder_saver = tf.train.Saver(var_list=tf.trainable_variables(self.embedder_model_scope))
      embedder_latest_checkpoint = tf.train.latest_checkpoint(self.embedder_dir)
      print("LOAD_STATE_ENCODER_EMBEDDER_CHECKPOINT: %s" % (embedder_latest_checkpoint,))
      embedder_saver.restore(self.sess, embedder_latest_checkpoint)
    if self.predictor_dir:
      predictor_saver = tf.train.Saver(var_list=tf.trainable_variables("^" + self.predictor_model_scope + "/"))
      print(tf.trainable_variables("^" + self.predictor_model_scope + "/"))
      predictor_latest_checkpoint = tf.train.latest_checkpoint(self.predictor_dir)
      print("LOAD_STATE_ENCODER_PREDICTOR_CHECKPOINT: %s" % (predictor_latest_checkpoint,))
      predictor_saver.restore(self.sess, predictor_latest_checkpoint)

  def encode(self, obses, actions):
    if self.embedder_dir:
      embedder_embeddings = self.sess.run(self.embedder_model_embeddings, { self.embedder_model_obses:np.expand_dims(obses[:,:,:,-1],-1) })
    else:
      embedder_embeddings = [ None for env_idx in range(len(obses)) ]
    if self.predictor_dir:
      predictor_eval_start_timestamp = datetime.datetime.now()
      eval_predictor_reward_values, eval_predictor_extra_values, predictor_train_global_step_value  = self.sess.run([ self.eval_predictor_rewards, self.eval_predictor_extras, self.predictor_train_global_step ], { self.eval_predictor_model_obses:obses, self.eval_predictor_actions:actions })
      predictor_eval_stop_timestamp = datetime.datetime.now()
      print("STATE_ENCODER_PREDICTOR_EVAL: timestamp=%s step=%s rewards=%s actions=%s extras=%s duration=%ssec" % (predictor_eval_stop_timestamp, self.predictor_eval_step, eval_predictor_reward_values, actions, eval_predictor_extra_values.tolist(), (predictor_eval_stop_timestamp - predictor_eval_start_timestamp).total_seconds()))
      predictor_rewards = eval_predictor_reward_values

      self.predictor_eval_step += 1

      self.predictor_replay_buffer.extend(zip(obses, actions))
      if len(self.predictor_replay_buffer) > self.predictor_replay_size:
        self.predictor_replay_buffer = self.predictor_replay_buffer[-self.predictor_replay_size:]

      if self.predictor_eval_step % self.predictor_batch_size == 0:
        predictor_train_start_timestamp = datetime.datetime.now()
        train_obses, train_actions = zip(*[random.choice(self.predictor_replay_buffer) for i in range(self.predictor_batch_size)])
        predictor_train_loss_value, _ = self.sess.run([self.predictor_train_loss, self.predictor_train_step], { self.train_predictor_model_obses:train_obses, self.train_predictor_actions:train_actions })
        predictor_train_stop_timestamp = datetime.datetime.now()
        print("STATE_ENCODER_PREDICTOR_TRAIN: timestamp=%s step=%s train_loss=%s duration=%ssec" % (predictor_train_stop_timestamp, self.predictor_eval_step, predictor_train_loss_value, (predictor_train_stop_timestamp - predictor_train_start_timestamp).total_seconds()))
    else:
      predictor_rewards = [ 0.0 for env_idx in range(len(obses)) ]
    return embedder_embeddings, predictor_rewards 

