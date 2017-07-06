import numpy as np
import tensorflow as tf

def object_mask_loss(vectors, mask_list, alpha, beta):
  """Computes loss
  Args:
    vectors: tensor,shape=(batch_size,x,y,embed_dim) giving vector ouput of CNN
    mask_list: list of tensors of shape=(batch_size,x,y,1) giving masks of
      objects in segmentation
    alpha: scalar parameter controlling strength between intra- and inter-object
      loss
  Returns:
    loss: scalar tensor giving loss
  """
  losses = []
  for mask in mask_list:
    losses.append(single_object_loss(vectors,mask,alpha,beta))

  loss = tf.divide(tf.add_n(losses),tf.constant(len(losses),
        dtype=tf.float32), name='loss')
  return loss

def single_object_loss(vectors, mask, alpha, beta):
  """Returns loss for a single object"""
  # Compute mean vector
  mean_vector = compute_object_mean(vectors, mask)

  # Compute affinity map
  af_map = affinity_map(vectors, mean_vector)

  # Compute cross entropy for this mask
  xe = bounded_cross_entropy(af_map, mask, alpha, beta)
  return tf.reduce_sum(xe)

def affinity_map(vectors, seed_vector):
  dif = tf.subtract(vectors, tf.reshape(seed_vector, [1,1,1,-1]))
  interaction = tf.reduce_sum(dif*dif, reduction_indices=[-1], keep_dims=True)
  return tf.exp(-0.5*interaction)

def bounded_cross_entropy(guess,truth,alpha,beta):
  guess = 0.999998*guess + 0.000001
  return  - beta * truth * tf.log(guess) - alpha*(1-truth) * tf.log(1-guess)

def compute_object_mean(vectors, mask):
  """Computes mean vector in object (specified by mask)
  Args:
    vectors: tensor of floats, shape=(batch_size,x,y,embed_dim)
    mask: binary tensor, shape=(batch_size,x,y,1)
    Returns:
    vector: tensor of floats, shape=(batch_size, embed_dim)
  """
  sum_vectors = tf.reduce_sum(tf.multiply(vectors, mask), [1,2])
  num_vectors = tf.reduce_sum(mask, [1,2])

  return tf.divide(sum_vectors, num_vectors)
