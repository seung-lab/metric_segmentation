import numpy as np
import tensorflow as tf

import loss

# Setup values
alpha = 1.0
mask = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.float32).reshape(1,4,4,1)
vectors = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.float32).reshape(1,4,4,1)
#vectors = np.ones((1,4,4,1),dtype=np.float32)

# Convert to tf tensors
alpha = tf.constant(alpha)
mask = tf.constant(mask)
vectors = tf.constant(vectors)
mask_list = [mask]*4


# Create tensors
l = loss.object_mask_loss(vectors, mask_list, alpha)

# Start session
sess = tf.Session()

# Run session()
x= sess.run(l)

# Display results
print(x)

