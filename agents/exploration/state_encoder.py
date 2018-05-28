import os
import datetime
import numpy as np
import tensorflow as tf
from exploration.autoencoder import autoencoder_model_scope, autoencoder_model

class StateEncoder:
  def __init__(self, sess, encoder_dir, reward_weight=float(os.environ.get('RETRO_STATE_ENCODER_REWARD_WEIGHT',0.0))):
    self.sess = sess
    self.encoder_dir = encoder_dir
    self.reward_weight = reward_weight
    with tf.variable_scope(autoencoder_model_scope):
      self.model_obses, (self.model_embeddings_original, _), self.reconstruction_errors, (self.reconstruction_loss, self.embedding_loss), self.train_loss = autoencoder_model(
                                                  use_noisy = os.environ.get('RETRO_STATE_ENCODER_NOISY', "false") == "true",
                                                  use_embedding_loss = os.environ.get('RETRO_STATE_ENCODER_EMBEDDING_LOSS', "false") == "true"
                                                )
    self.global_step = tf.train.create_global_step()
    self.train_step = tf.train.AdamOptimizer().minimize(self.train_loss, global_step = self.global_step)

  def initialize(self):
    saver = tf.train.Saver(var_list=tf.trainable_variables('autoencoder'))
    latest_checkpoint = tf.train.latest_checkpoint(self.encoder_dir)
    print("LOAD_STATE_ENCODER_CHECKPOINT: %s" % (latest_checkpoint,))
    saver.restore(self.sess, latest_checkpoint)

  def encode(self, obses):
    start_timestamp = datetime.datetime.now()
    model_embedding_original_values, reconstruction_errors_value, reconstruction_loss_value, embedding_loss_value = self.sess.run([self.model_embeddings_original, self.reconstruction_errors, self.reconstruction_loss, self.embedding_loss], { self.model_obses:obses })
    _, global_step_value = self.sess.run([self.train_step, self.global_step], { self.model_obses:obses })
    rewards = self.reward_weight * np.sqrt(reconstruction_errors_value / 84 / 84)
    stop_timestamp = datetime.datetime.now()
    print("STATE_ENCODER: timestamp=%s step=%s reconstruction_loss=%s embedding_loss=%s duration=%ssec" % (stop_timestamp, global_step_value, reconstruction_loss_value, embedding_loss_value, (datetime.datetime.now() - start_timestamp).total_seconds()))
    return model_embedding_original_values, rewards 

