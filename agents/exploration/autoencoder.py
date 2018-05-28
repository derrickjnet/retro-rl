import os
import tensorflow as tf

autoencoder_filters = int(os.environ.get("RETRO_AUTOENCODER_FILTERS",96))

if os.environ.get("RETRO_AUTOENCODER_ACTIVATION", "") == "sigmoid":
  default_autoencoder_embedding_activation = tf.nn.sigmoid
elif os.environ.get("RETRO_AUTOENCODER_ACTIVATION", "") == "relu": 
  default_autoencoder_embedding_activation = tf.nn.relu
else:
  assert False

def autoencoder_observations():
  with tf.variable_scope("observations"):
    observations = tf.placeholder(tf.uint8, [None, 84, 84, 1])
  return observations

def autoencoder_observations_rescaled(observations):
    with tf.variable_scope("rescaler"):
      observations_rescaled = tf.cast(observations, tf.float32) / 255.
    return observations_rescaled

def autoencoder_encoder(inputs, reuse=False, embedding_activation=None):
    if embedding_activation is None:
      embedding_activation = default_autoencoder_embedding_activation
    with tf.variable_scope("encoder", reuse=reuse):
      net = tf.layers.conv2d(inputs, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv1")
      net = tf.layers.conv2d(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv2")
      net = tf.layers.conv2d(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv3")
      net = tf.layers.conv2d(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv4")
      embeddings = tf.layers.dense(tf.reshape(net, [-1,6*6*autoencoder_filters]), 64, activation=embedding_activation, name="fc")
    return embeddings

def autoencoder_embeddings_noisy(embeddings):
    with tf.variable_scope("noise"):
      embeddings_noisy = embeddings + tf.random_uniform(tf.shape(embeddings), -0.3,0.3)
    return embeddings_noisy

def autoencoder_decoder(embeddings, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
      net = tf.layers.dense(embeddings, 6*6*autoencoder_filters, activation=tf.nn.relu, name="fc")
      net = tf.reshape(net, [-1, 6, 6, autoencoder_filters])
      net = tf.layers.conv2d_transpose(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv4")
      net = tf.layers.conv2d_transpose(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv3")
      net = tf.layers.conv2d_transpose(net, autoencoder_filters, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv2")                
      result = tf.layers.conv2d_transpose(net, 1, [6, 6], strides=2, padding='SAME', activation=tf.nn.sigmoid, name="deconv1")
      outputs = tf.image.resize_images(result, size=(84,84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return outputs

def autoencoder_reconstruction_errors(observations_rescaled, outputs):
  with tf.variable_scope("reconstruction"):
    reconstruction_errors = tf.reduce_sum(tf.square(observations_rescaled - outputs),[1,2,3])
  return reconstruction_errors

def autoencoder_embedding_loss(embeddings):
  with tf.variable_scope("embeddings"):
    embedding_loss = tf.reduce_mean(tf.reduce_sum(embeddings**2, [1]))
  return embedding_loss

autoencoder_model_scope = "autoencoder"

def autoencoder_model(use_noisy=False, use_embedding_loss=False, embedding_activation=None):
    model_obses = autoencoder_observations() 
    model_rescaled_obses = autoencoder_observations_rescaled(model_obses)
    model_embeddings_original = autoencoder_encoder(model_rescaled_obses, embedding_activation=embedding_activation)
    if use_noisy:
      model_embeddings = autoencoder_embeddings_noisy(model_embeddings_original)
    else:
      model_embeddings = model_embeddings_original
    model_outputs = autoencoder_decoder(model_embeddings)

    reconstruction_errors = autoencoder_reconstruction_errors(model_rescaled_obses, model_outputs)
    reconstruction_loss = tf.reduce_mean(reconstruction_errors)
    embedding_loss = autoencoder_embedding_loss(model_embeddings_original)
    if use_embedding_loss:
      train_loss = reconstruction_loss + embedding_loss
    else:
      train_loss = reconstruction_loss
    
    return model_obses, (model_embeddings_original, model_embeddings), reconstruction_errors, (reconstruction_loss, embedding_loss), train_loss
