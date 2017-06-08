"""Contains commonly used functions for analysis"""
import numpy as np
import tensorflow as tf
import h5py
import os


def load_model(model_name, sess):
  """Loads tensorflow model into session.
  Args:
    model_name: str, full path to model
    sess: tensorflow Session
  Returns:
    inputs: tensor, input em images to model
    outputs: tensor, output vector labels from model
  """
  saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True)
  saver.restore(sess, model_name)
  em_input = tf.get_default_graph().get_tensor_by_name("input_1:0")
  #vec_labels = tf.get_default_graph().get_tensor_by_name("conv2d_13/BiasAdd:0")
  vec_labels = tf.get_default_graph().get_tensor_by_name("conv2d_23/BiasAdd:0")
  return em_input, vec_labels

def generate_vector_labels(input_tensor, output_tensor, input_data, sess):
  """Generate vector labels for input_data
  Args:
    input_tensor: tensor, shape=(batch_size, x, y, 1), input placeholder
    output_tensor: tensor, shape=(batch_size, x, y, embed_dim), vector outputs from net 
    intput_data: np array, shape=(x,y), data to run net on
    sess: tf session with net loaded into memory
        
  Returns:
    output_data: numpy array, shape=(x,y,embed_dim), containing vector labels for input
  """
  # Reshape input data
  input_data = np.expand_dims(np.expand_dims(input_data, 0), -1)

  # Preprocess input data
  input_data = input_data - np.mean(input_data)
  
  # Compute vector labels
  output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})
  return output_data[0,:,:,:]
