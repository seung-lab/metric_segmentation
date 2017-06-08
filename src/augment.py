"""Provides some data augmentation"""
import numpy as np

def augment_example(imgs):
  """Augments example.
  Args:
    imgs: list of numpy arrays, each of shape-(x,y,1) of raw EM-image

  Returns:
    imgs: list of images after augmentation (note the same augmentation is
        applied to each image)
  """
  # Flips
  flips = (flip_x, flip_y)
  for flip in flips:
    if np.random.randint(0,2) == 1:
      imgs = [flip(img) for img in imgs]

  # Rotations
  rotations = (identity, rotate_90, rotate_180, rotate_270)
  rotate = np.random.choice(rotations)
  imgs = [rotate(img) for img in imgs]

  return imgs


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
