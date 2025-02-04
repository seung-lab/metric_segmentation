import numpy as np
import tensorflow as tf
import os

from .model import create_UNet
from .loss import object_mask_loss
from .provider import EMDataGenerator

def train(params):
  """Trains and monitors net"""
  data_dir = params['data_dir']
  log_dir = params['log_dir']
  model_dir = params['model_dir']
  save_dir = params['save_dir']
  K = params['n_sampled_objects']
  alpha = params['alpha']
  beta = params['beta']
  lr = params['learning_rate']
  
  # Create input, output placeholders
  em_input, vec_labels = create_UNet(params)
  mask_list_input = [tf.placeholder(dtype=tf.float32, shape=(1,params['out_height'], params['out_width'], 1))]*K

  # Create loss
  loss = object_mask_loss(vec_labels, mask_list_input,  alpha, beta)

  # Create train op
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)#, momentum=0.9)
  train_op = optimizer.minimize(loss)

  # Initialize data provider
  em_train = EMDataGenerator(data_dir, 'train', K)
  em_dev = EMDataGenerator(data_dir, 'dev', K)

  # Initialize session
  init_op = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init_op)

  # Create logging utils
  writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
  train_loss_summary = tf.summary.scalar("train_loss", loss, ['monitor'])
  valid_loss_summary = tf.summary.scalar("val_loss", loss)
  mask_summary_op = tf.summary.image("mask", mask_list_input[0],max_outputs=1)
  summary_op = tf.summary.merge_all('monitor')

  saver = tf.train.Saver(max_to_keep=100)

  # Train
  max_steps = 100000
  print("-------------Training-----------------")
  for i in range(max_steps):
    # Get Data
    em_input_data_train, mask_list_data_train = em_train.next_batch(1)
    em_input_data_dev, mask_list_data_dev = em_dev.next_batch(1)
    
    # Infer + Backprop
    phs = [em_input] + mask_list_input
    dats = [em_input_data_train] + mask_list_data_train
    feed_dict = dict((ph,dat) for ph,dat in zip(phs, dats))

    _, summary, mask_sum = sess.run([train_op, summary_op, mask_summary_op], feed_dict=feed_dict)
    #_, summary = sess.run([loss, summary_op], feed_dict=feed_dict)    
    l = sess.run(loss, feed_dict=feed_dict)    
    if l > 2E5:
      np.save('train_{}_{}'.format(i, l), mask_list_data_train)
    
    if i % 2 == 0:
      dats = [em_input_data_dev] + mask_list_data_dev
      feed_dict =dict((ph,dat) for ph,dat in zip(phs, dats))

      valid_summary = sess.run(valid_loss_summary, feed_dict=feed_dict)

      # Monitor progress
      writer.add_summary(summary, i)
      writer.add_summary(valid_summary, i)
      writer.add_summary(mask_sum, i)

    # Checkpoint periodically
    if i % 500 == 0:
      print("Processed {} epochs.".format(i))
      # Save model
      saver.save(sess, os.path.join(model_dir, "model{}.ckpt".format(i)))
