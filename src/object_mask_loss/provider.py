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
    assert(batch_size == 1)
    em, mask_list = self.next_example()

    # Reshape for batch size 1
    em_batch = np.expand_dims(em, 0)
    mask_list = [np.expand_dims(m,0) for m in mask_list]
    
    return em_batch, mask_list

  def next_example(self):
    """Returns next example and increments ptr
    example = (em_img, mask_list) where mask list is a list of binary masks
      containing objects """
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

    # create mask_list
    mask_list = self.create_mask_list(seg_img)

    # preprocess
    em_img = self.preprocess(em_img)

    # augment
    imgs = augment.augment_example([em_img]+mask_list)
    em_img = imgs[0]
    mask_list = imgs[1:]


    #import pdb; pdb.set_trace()
    # increment ptr
    self._ptr += 1

    return em_img, mask_list

  def create_mask_list(self, seg_img, K=10):
    """Creates list of masks for K randomly chosen objects
    Mask is numpy array of shape (x,y,1)
    """
    all_ids = np.unique(seg_img)
    chosen_ids = np.random.choice(all_ids, K)

    return [(seg_img == ID).astype(np.float32) for ID in chosen_ids]

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
    em_images = h5py.File(os.path.join(directory, "image.h5"), 'r')
    segmentations = h5py.File(os.path.join(directory, "human_labels.h5"), 'r')

    if group == 'train':
      return em_images['main'][:192], segmentations['main'][:192]
    elif group == 'dev':
      return em_images['main'][192:], segmentations['main'][192:]
