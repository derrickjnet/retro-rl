import sys
import os
import numpy as np
import tensorflow as tf
import time
import datetime

from exploration.autoencoder import autoencoder_model_scope, autoencoder_observations, autoencoder_observations_rescaled, autoencoder_encoder, autoencoder_embeddings_noisy, autoencoder_decoder, autoencoder_reconstruction_loss, autoencoder_embedding_loss

def parse_record(record_bytes, obs_steps=4):
  features = {
            'game_name' : tf.FixedLenFeature((), tf.string),
            'act_name' : tf.FixedLenFeature((), tf.string),
            'total_steps' : tf.FixedLenFeature((), tf.int64),
            'episode' : tf.FixedLenFeature((), tf.int64),
            'episode_step' : tf.FixedLenFeature((), tf.int64),
            'obs' : tf.FixedLenFeature((), tf.string)
          }
  result = tf.parse_single_example(record_bytes, features)
  obs = tf.reshape(tf.decode_raw(result['obs'], tf.uint8), [84,84,obs_steps])
  return (obs,)

def build_combined_dataset(events_path):
  return tf.data.Dataset.list_files(events_path + "*.events.tfrecords").apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=100, block_length=1)).map(parse_record, num_parallel_calls=10).shuffle(10000).repeat().batch(64).prefetch(1)

events_path = sys.argv[1]
output_path = sys.argv[2]

checkpoints_path = output_path + "tensorflow"
os.makedirs(checkpoints_path, exist_ok=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement=True
with tf.Session(config=config) as sess:
  with tf.variable_scope(autoencoder_model_scope):
    model_obs = autoencoder_observations() 
    model_rescaled_obs = autoencoder_observations_rescaled(model_obs)
    model_embeddings = autoencoder_encoder(model_rescaled_obs)
    if os.environ.get('RETRO_AUTOENCODER_NOISY', "false") == "true":
      model_embeddings = autoencoder_embeddings_noisy(model_embeddings) 
    model_outputs = autoencoder_decoder(model_embeddings)

    reconstruction_loss = autoencoder_reconstruction_loss(model_rescaled_obs, model_outputs)
    embedding_loss = autoencoder_embedding_loss(model_embeddings)
    train_loss = reconstruction_loss #+ embedding_loss

  saver = tf.train.Saver(var_list=tf.trainable_variables(autoencoder_model_scope), max_to_keep=None)

  dataset = build_combined_dataset(events_path) 
  batch_tensor = dataset.make_one_shot_iterator().get_next()

  global_step = tf.train.create_global_step()

  train_step = tf.train.AdamOptimizer().minimize(train_loss, global_step = global_step)

  tf.global_variables_initializer().run()

  latest_checkpoint = tf.train.latest_checkpoint(checkpoints_path)
  print("LOAD_CHECKPOINT: %s" % (latest_checkpoint,))
  if latest_checkpoint is not None:
    saver.restore(sess, latest_checkpoint)

  while True:
    (batch_obs,) = sess.run(batch_tensor)
    model_output_values, model_embedding_values, reconstruction_loss_value, embedding_loss_value, _, global_step_value = sess.run([model_outputs - model_rescaled_obs, model_embeddings, reconstruction_loss, embedding_loss, train_step, global_step], feed_dict={model_obs:np.expand_dims(batch_obs[:,:,:,-1],3)})
    print("STEP: timestamp=%s step=%s reconstruction_loss=%s embedding_loss=%s" % (datetime.datetime.now(), global_step_value, reconstruction_loss_value, embedding_loss_value))

    for model_embedding_value in model_embedding_values.tolist()[0:1]:
      print("EMBEDDING: %s" % (list(map(lambda v: round(v*10000)/10000.0,model_embedding_value)),))
      print("CODE: %s" % (list(map(lambda v: round(v),model_embedding_value)),))
    sys.stdout.flush()
    if global_step_value % 1000 == 0:
      print("SAVE_CHECKPOINT: step=%s" % (global_step_value,))
      saver.save(sess, checkpoints_path + "/checkpoint", global_step=int(time.time()))

