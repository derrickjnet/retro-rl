import sys
import os
import numpy as np
import tensorflow as tf
import time
import datetime

from exploration.autoencoder import Autoencoder 

def parse_record(record_bytes, obs_steps=4):
  features = {
            #'game_name' : tf.FixedLenFeature((), tf.string),
            #'act_name' : tf.FixedLenFeature((), tf.string),
            #'total_steps' : tf.FixedLenFeature((), tf.int64),
            #'episode' : tf.FixedLenFeature((), tf.int64),
            #'episode_step' : tf.FixedLenFeature((), tf.int64),
            'obs' : tf.FixedLenFeature((), tf.string)
          }
  result = tf.parse_single_example(record_bytes, features)
  obs = tf.reshape(tf.decode_raw(result['obs'], tf.uint8), [84,84,obs_steps])
  return (obs,)

def build_combined_dataset(events_path):
  return tf.data.Dataset.list_files(events_path + "*.events.tfrecords").apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=100, block_length=1)).map(parse_record, num_parallel_calls=10).shuffle(10000).repeat().batch(64).prefetch(1)

events_path = sys.argv[1]
output_path = sys.argv[2]

autoencoder_nfilters = int(os.environ["RETRO_AUTOENCODER_NFILTERS"])
autoencoder_embedding_size = int(os.environ["RETRO_AUTOENCODER_EMBEDDING_SIZE"])
autoencoder_use_noisy = os.environ['RETRO_AUTOENCODER_NOISY'] == "true"
autoencoder_use_embedding_loss = os.environ['RETRO_AUTOENCODER_EMBEDDING_LOSS'] == "true"
autoencoder_model_scope = os.environ['RETRO_AUTOENCODER_MODEL_SCOPE']

print("AUTOENCODER_PARAMS: nfilters=%s embedding_size=%s use_noisy=%s use_embedding_loss=%s mode_scope=%s" % (autoencoder_nfilters, autoencoder_embedding_size, autoencoder_use_noisy, autoencoder_use_embedding_loss, autoencoder_model_scope))

checkpoints_path = output_path + "tensorflow"
os.makedirs(checkpoints_path, exist_ok=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement=True
with tf.Session(config=config) as sess:
  with tf.variable_scope(autoencoder_model_scope):
    autoencoder = Autoencoder(nfilters = autoencoder_nfilters, embedding_size = autoencoder_embedding_size)
    model_obs, (model_embeddings_original, model_embeddings), _, (reconstruction_loss, embedding_loss), train_loss = autoencoder.model(
                                                  use_noisy = autoencoder_use_noisy, 
                                                  use_embedding_loss = autoencoder_use_embedding_loss 
                                                )

  saver = tf.train.Saver(var_list=tf.trainable_variables(autoencoder_model_scope), max_to_keep=None)

  dataset = build_combined_dataset(events_path) 
  batch_tensor = dataset.make_one_shot_iterator().get_next()

  global_step = tf.train.create_global_step()

  train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(train_loss, global_step = global_step)

  tf.global_variables_initializer().run()

  latest_checkpoint = tf.train.latest_checkpoint(checkpoints_path)
  print("LOAD_CHECKPOINT: %s" % (latest_checkpoint,))
  if latest_checkpoint is not None:
    saver.restore(sess, latest_checkpoint)

  while True:
    (batch_obs,) = sess.run(batch_tensor)
    model_embedding_original_values, model_embedding_values, reconstruction_loss_value, embedding_loss_value, train_loss_value, _, global_step_value = sess.run([model_embeddings_original, model_embeddings, reconstruction_loss, embedding_loss, train_loss, train_step, global_step], feed_dict={model_obs:np.expand_dims(batch_obs[:,:,:,-1],3)})
    print("STEP: timestamp=%s step=%s reconstruction_loss=%s embedding_loss=%s train_loss=%s" % (datetime.datetime.now(), global_step_value, reconstruction_loss_value, embedding_loss_value, train_loss_value))

    for model_embedding_original_value in model_embedding_original_values.tolist()[0:1]:
      print("EMBEDDING: %s" % (list(map(lambda v: round(v*10000)/10000.0,model_embedding_original_value)),))
    sys.stdout.flush()
    if global_step_value % 1000 == 0:
      print("SAVE_CHECKPOINT: step=%s" % (global_step_value,))
      saver.save(sess, checkpoints_path + "/checkpoint", global_step=int(time.time()))

