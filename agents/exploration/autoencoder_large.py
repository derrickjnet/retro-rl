import tensorflow as tf

def encoder_rescaled(inputs):
    inputs = tf.cast(inputs, tf.float32) / 255.
    return inputs, encoder(inputs)

def encoder(inputs, reuse=False):
    with tf.variable_scope("autoencoder/encoder", reuse=reuse):
      net = tf.layers.conv2d(inputs, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv1")
      net = tf.layers.conv2d(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv2")
      net = tf.layers.conv2d(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv3")
      net = tf.layers.conv2d(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="conv4")
      embeddings = tf.layers.dense(tf.reshape(net, [-1,6*6*96]), 64, activation=tf.nn.sigmoid, name="fc")
    return embeddings

def decoder(embeddings, reuse=False):
    with tf.variable_scope("autoencoder/decoder", reuse=reuse):
      net = tf.layers.dense(embeddings, 6*6*96, activation=tf.nn.relu, name="fc")
      net = tf.reshape(net, [-1, 6, 6, 96])
      net = tf.layers.conv2d_transpose(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv4")
      net = tf.layers.conv2d_transpose(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv3")
      net = tf.layers.conv2d_transpose(net, 96, [6, 6], strides=2, padding='SAME', activation=tf.nn.relu, name="deconv2")                
      result = tf.layers.conv2d_transpose(net, 1, [6, 6], strides=2, padding='SAME', activation=tf.nn.sigmoid, name="deconv1")
      outputs = tf.image.resize_images(result, size=(84,84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return outputs

