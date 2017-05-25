"""Contains function to create UNet
reference: https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py
"""
import keras as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D,\
            Cropping2D, UpSampling2D
from keras.layers.merge import Concatenate

def create_UNet(params):
  """Creates UNet using Keras Functional API

  Returns:
    input: input tensor
    output: output tensor
  """
  height, width = params['height'], params['width']
  embed_dim = params['embed_dim']
  # Create layers
  inputs = Input(shape=(height, width, 1))

  conv0 = Convolution2D(64, (3, 3), activation='relu', padding='valid')(inputs)
  conv1 = Convolution2D(64, (3, 3), activation='relu', padding='valid')(conv0)
  conv2 = downsample_block(conv1, 128)
  conv3 = downsample_block(conv2, 256)
  conv4 = downsample_block(conv3, 512)
  conv5 = downsample_block(conv4, 1024)

  up1 = upsample_block(conv4, conv5, 512, (4,4))
  up2 = upsample_block(conv3, up1, 256, (16,16))
  up3 = upsample_block(conv2, up2, 128, (40,40))
  up4 = upsample_block(conv1, up3, 64, (88,88))

  outputs = Convolution2D(embed_dim, (1, 1), padding='same')(up4)

  return inputs, outputs

def downsample_block(inputs, filters):
  """Returns a layer implementing the core downsample operation in UNet
      maxpool-conv-relu-conv-relu

  Args:
    inputs: input tensor to block
    filters: integer, number of activation units per layer in block

  Returns:
    outputs: output tensor of layer
  """
  pool = MaxPooling2D(pool_size=(2,2))(inputs)
  conv = Convolution2D(filters, (3, 3), activation='relu', padding='valid')(pool)
  conv = Convolution2D(filters, (3, 3), activation='relu', padding='valid')(conv)
  return conv

def upsample_block(side_inputs, up_inputs, filters, crop):
  """Returns a layer implementing the core upsample operation in UNet

  Args:
    side_inputs: input tensor to copy and crop
    up_inputs: input tensor to upsample
    filters: integer, number of activation units per layer in block

  Returns:
    outputs: output tensor of layer
  """
  up_a = Cropping2D(cropping=crop)(side_inputs)
  up_b = UpSampling2D(size=(2,2))(up_inputs)
  up = Concatenate()([up_a, up_b])

  conv = Convolution2D(filters, (3, 3), activation='relu', padding='valid')(up)
  conv = Convolution2D(filters, (3, 3), activation='relu', padding='valid')(conv)

  return conv
