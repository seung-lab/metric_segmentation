import numpy as np
import h5py
import os

import augment

class EMDataGenerator:
  """Generator for batches of EM images and their segmentations.
  example:
    >>> gen = EMDataGenerator(target_size, directory)
    >>> for em_img, segment_img in gen.flow(batch_size):
    >>>   process_batch(X,y)
  """
  def __init__(self, directory, group, augment=False):
    """
    Args:
      directory: path containing data. See DirectoryIterator.exampleMap for more
      group: 'train' or 'dev'
      augment: augment returned batches
    """
    self.augment = augment

    self.em_data, self.seg_data = self.load_data(directory, group)
    self._ptr = 0

  def next_batch(self, batch_size):
    """Return batch of images and their segmentations"""
    # Get batch
    em_batch = []
    seg_batch = []
    mask_batch = []
    for i in range(batch_size):
      em, seg, mask = self.next_example()
      em_batch.append(em)
      seg_batch.append(seg)
      mask_batch.append(mask)

    em_batch = np.array(em_batch)
    seg_batch = np.array(seg_batch)
    mask_batch = np.array(mask_batch)

    # Preprocess examples
    # yield batch
    return em_batch, seg_batch, mask_batch

  def next_example(self):
    """Returns next example and increments ptr"""
    if self._ptr == len(self.em_data):
      self._ptr = 0

    i = self._ptr
      
    # select slice
    em_img = self.em_data[i]
    seg_img = self.seg_data[i]

    # crop region
    dx = np.random.randint(0,1024-572)
    dy = np.random.randint(0,1024-572)

    em_img = em_img[dx:dx+572, dy: dy+572]
    seg_img = seg_img[dx:dx+572, dy: dy+572]
    l = (572-388)//2
    u = 388+l
    seg_img = seg_img[l:u, l:u]

    # add channel dimension
    em_img = np.expand_dims(em_img, axis=-1)
    seg_img = np.expand_dims(seg_img, axis=-1)

    # create mask
    mask_img = np.ones_like(seg_img)
    mask_img[seg_img==0] = 1

    # preprocess
    em_img = self.preprocess(em_img)

    # augment
    em_img, seg_img = augment.augment_example([em_img, seg_img])
    
    # increment ptr
    self._ptr += 1

    return em_img, seg_img, mask_img

  def preprocess(self, img):
    """Preprocess img, does mean subtraction

    Args:
      img: shape (x,y,channel)

    Returns:
      img: processed shape (x,y,channel)
    """
    return img - np.mean(img)


  def load_data(self, directory, group):
    """Loads data, EM images and their segmentations, from directory
    Args:
      directory: path to dir containing:
                    human_labels.h5: segmentation image
                    image.h5: raw EM image
      group: 'train' or 'dev'

    Returns:
      (em_images, segmentations):
        em_images: np array (n_images, x, y)
        segmentations: np array (n_images, x ,y)
    """
    #em_images = h5py.File(os.path.join(directory, "human_labels_boundary.h5"), 'r')
    #segmentations = h5py.File(os.path.join(directory, "human_labels_split.h5"), 'r')
    """
    if group == 'train':
      return em_images['main'][:192], segmentations['main'][:192]
    elif group == 'dev':
      return em_images['main'][192:], segmentations['main'][192:]
    """
    em_images = h5py.File(os.path.join(directory, "voronoi_boundary.h5"), 'r')
    segmentations = h5py.File(os.path.join(directory, "voronoi_boundary.h5"), 'r')

    if group == 'train':
      return em_images['main'][:16], segmentations['main'][:16]
    elif group == 'dev':
      return em_images['main'][16:], segmentations['main'][16:]
              
