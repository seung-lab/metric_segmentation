import h5py
import numpy as np
import tensorflow as tf

from model import create_UNet
from provider import EMDataGenerator

# Load em images for train and dev
# Load saved model
# Run this model on each and save results
# Save h5
def crop(img):
  target_size = 388
  input_size = 572
  l = (input_size-target_size)//2
  u = l + target_size
  return img[l:u, l:u]


# Load imgs
direc = "/usr/people/kluther/Documents/metric_segmentation/data"
em = EMDataGenerator(direc, 'train')
em_dev = EMDataGenerator(direc, 'dev')
print("-------Loaded Images----------")

# Create input and vec_labels and restore model
version = 70000
saver = tf.train.import_meta_graph("./saved/model{}.ckpt.meta".format(version), clear_devices=True)
sess = tf.Session(config=tf.ConfigProto(device_count={"GPU": 0, "CPU": 1}))

em_input = tf.get_default_graph().get_tensor_by_name("input_1:0")
vec_labels = tf.get_default_graph().get_tensor_by_name("conv2d_23/BiasAdd:0")

#saver = tf.train.Saver()
#sess = tf.Session()

model_name = "./saved/model{}.ckpt".format(version)
saver.restore(sess, model_name)
print("----------Restored Session----------")

# For each images, run and store
train_inputs = []
train_output_vecs = []
train_output_ids = []
for i in range(192*4):
  # Run
  em_input_data, human_label_data = em.next_batch(1)
  feed_dict = {em_input: em_input_data}
  vec_label_data = sess.run(vec_labels, feed_dict=feed_dict)

  # Store
  train_inputs.append(crop(em_input_data[0,:,:,0]))
  train_output_vecs.append(vec_label_data[0,:,:])
  train_output_ids.append(human_label_data[0,:,:,0])

  if i % 10 == 0:
    print("Ran {} training iter".format(i))
  if i == 20: break

train_inputs = np.array(train_inputs)
train_output_vecs = np.array(train_output_vecs)
train_output_ids = np.array(train_output_ids)

dev_inputs = []
dev_output_vecs = []
dev_output_ids = []
for i in range((256-192)*4):
  # Run
  em_input_data, human_label_data = em.next_batch(1)
  feed_dict = {em_input: em_input_data}
  vec_label_data = sess.run(vec_labels, feed_dict=feed_dict)

  # Store
  dev_inputs.append(crop(em_input_data[0:,:,0]))
  dev_output_vecs.append(vec_label_data[0,:,:])
  dev_output_ids.append(human_label_data[0,:,:,0])

  if i == 10: break

dev_inputs = np.array(dev_inputs)
dev_output_vecs = np.array(dev_output_vecs)
dev_output_ids = np.array(dev_output_ids)

# Save
train_h5 = h5py.File('train.hdf5','w')
train_h5.create_dataset('em', data=train_inputs)
train_h5.create_dataset('vecs', data=train_output_vecs)
train_h5.create_dataset('ids', data=train_output_ids)
train_h5.close()

valid_h5 = h5py.File('valid.hdf5','w')
valid_h5.create_dataset('em', data=dev_inputs)
valid_h5.create_dataset('vecs', data=dev_output_vecs)
valid_h5.create_dataset('ids', data=dev_output_ids)
valid_h5.close()
