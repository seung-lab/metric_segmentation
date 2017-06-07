"""Provides some data augmentation"""
import numpy as np

def augment_example(em_img, seg_img):
  """Augments example.
  Args:
    em_img: numpy array, shape-(x,y,1) of raw EM-image
    seg_img: numpy array, shape-(x,y,1) of segmentations (integer ids)

  Returns:
    em_img: after augmentation
    seg_img: after augmentation
  """
  # Flips
  flips = (flip_x, flip_y)
  for flip in flips:
    if np.random.randint(0,2) == 1:
      em_img = flip(em_img)
      seg_img = flip(seg_img)

  # Rotations
  rotations = (identity, rotate_90, rotate_180, rotate_270)
  rotate = np.random.choice(rotations)

  em_img = rotate(em_img)
  seg_img = rotate(seg_img)

  return em_img, seg_img


def flip_x(img):
  return np.flip(img, axis=0)
def flip_y(img):
  return np.flip(img, axis=1)

def identity(img):
  return img
def rotate_90(img):
  return np.rot90(img, k=1, axes=(0,1))
def rotate_180(img):
  return np.rot90(img, k=2, axes=(0,1))
def rotate_270(img):
  return np.rot90(img, k=3, axes=(0,1))
