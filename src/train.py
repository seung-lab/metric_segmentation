import numpy as np
import tensorflow as tf

from model import create_UNet
from loss import long_range_loss_fun
from provider import EMDataGenerator

# Create params of graph
params = {'height': 572, 'width': 572, 'embed_dim': 32}

# Create input, output placeholders
em_input, vec_labels = create_UNet(params)
mask = tf.constant(np.ones((1, params['height'], params['width'], 1)), dtype=tf.float32)
human_labels = tf.placeholder(dtype=tf.int32, shape=(1,params['height'], params['width'], 1))

# Create loss tensor
offsets = [(0,0,5), (0,5,0), (5,0,0)]
loss, outputs = long_range_loss_fun(vec_labels, human_labels, offsets, mask)

# Create train op
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# Initialize data provider
em = EMDataGenerator("/usr/people/kluther/seungmount/research/datasets/SNEMI3D/AC3")

# Initialize session
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Create logging utils
writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
loss_summary = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

# Train
max_steps = 1000
for i in range(max_steps):
  em_input_data, human_label_data = em.next_batch(1)
  feed_dict = {em_input: em_input_data, human_labels: human_label_data}

  _, summary = sess.run([train_op, summary_op], feed_dict=feed_dict)

  # Checkpoint and monitor
  writer.add_summary(summary, i)

  if i % 10 == 0:
    saver.save(sess, "./saved/model.ckpt")
    vec_label_data = sess.run(vec_labels, feed_dict=feed_dict)
    np.save('saved/vec_labels', vec_label_data)
    np.save('saved/human_labels', human_label_data)
