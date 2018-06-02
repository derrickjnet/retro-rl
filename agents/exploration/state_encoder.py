import os
import datetime
import numpy as np
import tensorflow as tf
from exploration.autoencoder import Autoencoder 

class StateEncoder:
  def __init__(self, sess, num_actions, embedder_dir=os.environ.get('RETRO_STATE_EMBEDDER_DIR'), embedder_model_scope=os.environ.get('RETRO_STATE_EMBEDDER_MODEL_SCOPE', 'state_embedder'), embedder_nfilters=int(os.environ.get('RETRO_STATE_EMBEDDER_NFILTERS',32)), embedder_embedding_size=int(os.environ.get('RETRO_STATE_EMBEDDER_EMBEDDING_SIZE',256)), predictor_dir=os.environ.get('RETRO_STATE_PREDICTOR_DIR'), predictor_model_scope=os.environ.get('RETRO_STATE_PREDICTOR_MODEL_SCOPE', 'state_predictor'), predictor_nfilters=int(os.environ.get('RETRO_STATE_PREDICTOR_NFILTES', 32)), predictor_embedding_size=int(os.environ.get('RETRO_STATE_PREDICTOR_EMBEDDING_SIZE', 256))):
    self.sess = sess
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
      with tf.variable_scope(predictor_model_scope, reuse=tf.AUTO_REUSE):
        self.predictor_actions = tf.placeholder(tf.int32, [None])
        self.predictor_model_obses1, _, self.predictor_model_embeddings1 = self.predictor.embed(embedding_activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE) 
        self.predictor_model_obses2, _, self.predictor_model_embeddings2 = self.predictor.embed(embedding_activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
      with tf.variable_scope(predictor_model_scope + "_internal"):
        predictor_net = tf.layers.dense(tf.concat([self.predictor_model_embeddings1, self.predictor_model_embeddings2], -1), 256, activation=tf.nn.relu, name="fc1")
        self.predictor_logits = tf.layers.dense(predictor_net, num_actions, name="fc2")
        self.predictor_probs = tf.nn.softmax(self.predictor_logits,-1)
        self.predictor_inverse_errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictor_logits, labels=self.predictor_actions)
        self.predictor_train_loss = tf.reduce_mean(self.predictor_inverse_errors)
        self.predictor_global_step = tf.train.create_global_step()
        self.predictor_train_step = tf.train.AdamOptimizer().minimize(self.predictor_train_loss, global_step = self.predictor_global_step)

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
      predictor_start_timestamp = datetime.datetime.now()
      predictor_prob_values, predictor_inverse_errors_value, _, predictor_global_step_value = self.sess.run([self.predictor_probs, self.predictor_inverse_errors, self.predictor_train_step, self.predictor_global_step], { self.predictor_model_obses1:np.expand_dims(obses[:,:,:,-1],-1), self.predictor_model_obses2:np.expand_dims(obses[:,:,:,-2],-1), self.predictor_actions:actions })
      predictor_rewards = predictor_inverse_errors_value
      predictor_stop_timestamp = datetime.datetime.now()
      print("STATE_ENCODER_PREDICTOR: timestamp=%s step=%s inverse_errors=%s actions=%s action_probs=%s duration=%ssec" % (predictor_stop_timestamp, predictor_global_step_value, predictor_inverse_errors_value, actions, list([list(ap) for ap in predictor_prob_values]), (predictor_stop_timestamp - predictor_start_timestamp).total_seconds()))
    else:
      predictor_rewards = [ 0.0 for env_idx in range(len(obses)) ]
    return embedder_embeddings, predictor_rewards 

