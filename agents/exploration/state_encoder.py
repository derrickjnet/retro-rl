import numpy as np
import tensorflow as tf
from exploration.autoencoder import encoder_rescaled

class StateEncoder:
  def __init__(self, sess, encoder_dir):
    self.sess = sess
    self.autoencoder_obs = tf.placeholder(tf.uint8, [None, 84, 84, 1])
    self.autoencoder_embeddings = encoder_rescaled(self.autoencoder_obs)[1]
    self.encoder_dir = encoder_dir

  def initialize(self):
    saver = tf.train.Saver(var_list=tf.trainable_variables('autoencoder'))
    latest_checkpoint = tf.train.latest_checkpoint(self.encoder_dir)
    print("LOAD_AUTOENCODER_CHECKPOINT: %s" % (latest_checkpoint,))
    saver.restore(self.sess, latest_checkpoint)

  def encode(self, obses):
    return self.sess.run(self.autoencoder_embeddings, obses)

