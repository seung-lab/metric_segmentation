import numpy as np
import tensorflow as tf

def object_mask_loss(vectors, mask_list, seed_list, alpha):
  """Computes loss
  Args:
    vectors: tensor,shape=(batch_size,x,y,embed_dim) giving vector ouput of CNN
    mask_list: list of tensors of shape=(batch_size,x,y,1) giving masks of
      objects in segmentation
    seed_list: list of coordinates giving points to compute affinities from
    alpha: scalar parameter controlling strength between intra- and inter-object
      loss
  Returns:
    loss: scalar tensor giving loss
  """
  losses = []
  for mask, seed in zip(mask_list, seed_list):
    losses.append(single_object_loss(vectors,mask,seed,alpha))

  loss = tf.add_n(losses, name='loss')
  return loss 
                  
def single_object_loss(vectors, mask, seed, alpha):
  """Returns loss for a single object"""
  # Compute affinity map
  x = seed[0]
  y = seed[1]
  seed_vector = vectors[0,x,y,:]
  af_map = affinity_map(vectors, seed_vector)

  # Compute cross entropy for this mask
  xe = bounded_cross_entropy(af_map, mask, alpha)
  return tf.reduce_sum(xe)

def affinity_map(vectors, seed_vector):
  dif = tf.subtract(vectors, tf.reshape(seed_vector, [1,1,1,-1]))
  interaction = tf.reduce_sum(dif*dif, reduction_indices=[-1], keep_dims=True)
  return tf.exp(-0.5*interaction)

def bounded_cross_entropy(guess,truth,alpha):
  guess = 0.999998*guess + 0.000001
  return  - truth * tf.log(guess) - alpha*(1-truth) * tf.log(1-guess)
