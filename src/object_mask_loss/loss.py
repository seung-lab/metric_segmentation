import numpy as np
import tensorflow as tf

def object_mask_loss(vectors, mask_list, alpha=1.0):
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
  # Compute mean vector for each object
  mean_vectors = [compute_object_mean(vectors, mask) for mask in mask_list]
  [tf.summary.histogram('v{}'.format(i), v, collections=['monitor']) for i,v in enumerate(mean_vectors)]
  
  # Compute variance of vectors for each object
  intra_variances = [compute_object_variance(vectors, mask, mean) for mask, mean in \
                zip(mask_list, mean_vectors)]
  intra_variance = tf.add_n(intra_variances, name='intra_variance')
  tf.summary.scalar('intra_variance', intra_variance, collections=['monitor'])
  
  # Compute variance between means of vectors
  inter_variance = compute_vector_variance(mean_vectors, name='inter_variance')
  tf.summary.scalar('inter_variance', inter_variance, collections=['monitor'])
  
  # Compute loss
  loss = tf.add(intra_variance, tf.multiply(-alpha, inter_variance), name='loss')

  return loss

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

def compute_object_variance(vectors, mask, mean_vector):
  """Computes variance of masked region of vectors
  Args:
    vectors: tensor of floats, shape=(batch_size,x,y,embed_dim)
    mask: binary tensor, shape=(batch_size,x,y,1)
    mean_vector: tensor of floats, shape=(batch_size,1,1,embed_dim). This is
      the mean vector for the masked region
  Returns:
    variance: scalar tensor
  """
  mean_vector = tf.expand_dims(tf.expand_dims(mean_vector,1),1)
  centered_vectors = tf.subtract(vectors, mean_vector)
  variances = tf.square(centered_vectors)
  masked_variances = tf.multiply(variances, mask)

  sum_variances = tf.reduce_sum(masked_variances)
  num_variances = tf.reduce_sum(mask)

  return tf.divide(sum_variances, num_variances)

def compute_vector_variance(vector_list, name=None):
  """Computes variance of vector_list
  Args:
    vector_list: list of tensors of shape (batch_size, embed_dim). These
      tensors are the mean vectors for each object. Therefore, the length of
      this list equals the number of objects
  Returns:
    variance: scalar tensor
  """
  mean_vector = tf.divide(tf.add_n(vector_list), len(vector_list))
  centered_vectors = [tf.subtract(v, mean_vector) for v in vector_list]
  centered_vectors_sq = [tf.square(v) for v in centered_vectors]
  variance = tf.divide(tf.reduce_sum(tf.add_n(centered_vectors_sq)), tf.cast(len(vector_list),tf.float32), name=name)

  return variance
