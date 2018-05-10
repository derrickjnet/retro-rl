import sys
import os
import numpy as np
import tensorflow as tf
import time

#import baselines.ppo2.policies as policies
import ppo2.policies as policies
from gym import spaces

def build_dataset(data_file_name):
  return tf.data.TFRecordDataset(data_file_name).shuffle(10000).repeat()

def parse_record(record_bytes, obs_steps=4, target_count=7):
  features = {
            'game_name' : tf.FixedLenFeature((), tf.string),
            'act_name' : tf.FixedLenFeature((), tf.string),
            'total_steps' : tf.FixedLenFeature((), tf.int64),
            'episode' : tf.FixedLenFeature((), tf.int64),
            'episode_step' : tf.FixedLenFeature((), tf.int64),
            'obs' : tf.FixedLenFeature((), tf.string),
            'targets' : tf.FixedLenFeature((target_count), tf.float32),
            'action' : tf.FixedLenFeature((), tf.int64) 
          }
  result = tf.parse_single_example(record_bytes, features) 
  obs = tf.reshape(tf.decode_raw(result['obs'], tf.uint8), [84,84,obs_steps])
  targets = result['targets']
  return (obs, targets)

def build_combined_dataset(data_file_path):
  return tf.data.Dataset.list_files(data_file_path + "*.events.tfrecords").interleave(build_dataset, cycle_length=100, block_length=1).map(parse_record, num_parallel_calls=10).batch(64).prefetch(1)

data_file_path = sys.argv[1]

checkpoints_path = data_file_path + "/tensorflow/"
os.makedirs(checkpoints_path, exist_ok=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement=True
with tf.Session(config=config) as sess:
  train_model = policies.CnnPolicy(sess, np.zeros([84,84,4]), spaces.Discrete(7), 64, 4, reuse=False)
  saver = tf.train.Saver(var_list=tf.trainable_variables('ppo2_model'), max_to_keep=None)
  latest_checkpoint = tf.train.latest_checkpoint(checkpoints_path)
  if latest_checkpoint is not None:
    saver.restore(sess, latest_checkpoint)

  dataset = build_combined_dataset(data_file_path) 
  batch_tensor = dataset.make_one_shot_iterator().get_next()

  global_step = tf.train.create_global_step()

  train_targets = tf.placeholder(tf.float32, [None, 7])
  train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_targets, logits=train_model.pd.logits))
  train_step = tf.train.AdamOptimizer().minimize(train_loss, global_step = global_step)

  tf.global_variables_initializer().run()

  while True:
    (batch_obs, batch_targets) = sess.run(batch_tensor)
    train_loss_value, _, global_step_value = sess.run([train_loss, train_step, global_step], feed_dict={train_model.X:batch_obs, train_targets:batch_targets})
    print("STEP: step=%s loss=%s" % (global_step_value, train_loss_value))
    sys.stdout.flush()
    if global_step_value % 1000 == 0:
      print("CHECKPOINT: step=%s" % (global_step_value,))
      saver.save(sess, checkpoints_path + "/checkpoint", global_step=int(time.time()))

