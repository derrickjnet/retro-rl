import os
import datetime
import numpy as np
import tensorflow as tf
from exploration.autoencoder import Autoencoder 

class StateEncoder:
  def __init__(self, sess, embedder_dir=os.environ['RETRO_STATE_EMBEDDER_DIR'], embedder_model_scope=os.environ.get('RETRO_STATE_EMBEDDER_MODEL_SCOPE', 'state_embedder'), embedder_nfilters=int(os.environ.get('RETRO_STATE_EMBEDDER_NFILTERS',32)), predictor_dir=os.environ.get('RETRO_STATE_PREDICTOR_DIR'), predictor_model_scope=os.environ.get('RETRO_STATE_PREDICTOR_MODEL_SCOPE', 'state_predictor'), predictor_nfilters=int(os.environ.get('RETRO_STATE_PREDICTOR_NFILTES', 32)), predictor_reward_weight=float(os.environ.get('RETRO_STATE_PREDICTOR_REWARD_WEIGHT',1.0))):
    self.sess = sess
    self.embedder_dir = embedder_dir
    self.predictor_dir = predictor_dir
    if self.embedder_dir:
      self.embedder = Autoencoder(nfilters=embedder_nfilters)
      self.embedder_model_scope = embedder_model_scope
      with tf.variable_scope(embedder_model_scope):
        self.embedder_model_obses = self.embedder.observations() 
        self.embedder_model_rescaled_obses = self.embedder.observations_rescaled(self.embedder_model_obses)
        self.embedder_model_embeddings = self.embedder.encoder(self.embedder_model_rescaled_obses, embedding_activation=tf.nn.sigmoid)
  
    if self.predictor_dir:
      self.predictor_reward_weight = predictor_reward_weight
      self.predictor = Autoencoder(nfilters=predictor_nfilters)
      self.predictor_model_scope = predictor_model_scope
      with tf.variable_scope(predictor_model_scope):
        self.predictor_model_obses, _, self.predictor_reconstruction_errors, (self.predictor_reconstruction_loss, self.predictor_embedding_loss), self.predictor_train_loss = self.predictor.model(use_noisy=False, use_embedding_loss=True)
        self.predictor_global_step = tf.train.create_global_step()
        self.predictor_train_step = tf.train.AdamOptimizer().minimize(self.predictor_train_loss, global_step = self.predictor_global_step)

  def initialize(self):
    if self.embedder_dir:
      embedder_saver = tf.train.Saver(var_list=tf.trainable_variables(self.embedder_model_scope))
      embedder_latest_checkpoint = tf.train.latest_checkpoint(self.embedder_dir)
      print("LOAD_STATE_ENCODER_EMBEDDER_CHECKPOINT: %s" % (embedder_latest_checkpoint,))
      embedder_saver.restore(self.sess, embedder_latest_checkpoint)
    if self.predictor_dir:
      predictor_saver = tf.train.Saver(var_list=tf.trainable_variables(self.predictor_model_scope))
      predictor_latest_checkpoint = tf.train.latest_checkpoint(self.predictor_dir)
      print("LOAD_STATE_ENCODER_PREDICTOR_CHECKPOINT: %s" % (predictor_latest_checkpoint,))
      predictor_saver.restore(self.sess, predictor_latest_checkpoint)

  def encode(self, obses):
    if self.embedder_dir:
      embedder_embeddings = self.sess.run(self.embedder_model_embeddings, { self.embedder_model_obses:obses })
    else:
      embedder_embeddings = [ None for env_idx in range(len(obses)) ]
    if self.predictor_dir:
      predictor_start_timestamp = datetime.datetime.now()
      predictor_reconstruction_errors_value, predictor_reconstruction_loss_value, predictor_embedding_loss_value, _, predictor_global_step_value = self.sess.run([self.predictor_reconstruction_errors, self.predictor_reconstruction_loss, self.predictor_embedding_loss, self.predictor_train_step, self.predictor_global_step], { self.predictor_model_obses:obses })
      _, predictor_global_step_value = self.sess.run([self.predictor_train_step, self.predictor_global_step], { self.predictor_model_obses:obses })
      predictor_rewards = self.predictor_reward_weight * np.sqrt(predictor_reconstruction_errors_value / 84 / 84)
      predictor_stop_timestamp = datetime.datetime.now()
      print("STATE_ENCODER_PREDICTOR: timestamp=%s step=%s reconstruction_loss=%s embedding_loss=%s duration=%ssec" % (predictor_stop_timestamp, predictor_global_step_value, predictor_reconstruction_loss_value, predictor_embedding_loss_value, (predictor_stop_timestamp - predictor_start_timestamp).total_seconds()))
    else:
      predictor_rewards = [ 0.0 for env_idx in range(len(obses)) ]
    return embedder_embeddings, predictor_rewards 

