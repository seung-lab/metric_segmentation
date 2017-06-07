N = 6000

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


thresh = 0.4

# Load training imgs
v0 = np.load('vec_labels{}.npy'.format(N))[0]
v0 = np.linalg.norm(v0, axis=-1)

h0 = np.load('human_labels{}.npy'.format(N))[0,:,:,0]
l = (572-388)//2
u = 388+l

e0 = np.load('em_img{}.npy'.format(N))[0,l:u, l:u,0]

v100 = np.load('vec_labels{}.npy'.format(N))[0]
v100c = v100[:,:,0:3]
h100 = np.load('human_labels{}.npy'.format(N))[0,:,:,0]

d100 = np.load('vec_labels_dev{}.npy'.format(N))[0]
print(d100[100,100,:])

#d100 = np.load('em_img_dev{}_0.npy'.format(N))[0]
d100c = d100[:,:,0:3]

de100 = np.load('em_img_dev{}.npy'.format(N))[0,l:u,l:u,0]


def gradient_norm(vec_field):
  """Returns norm of gradient at each point in vec_field"""
  gx = vec_field[1:,:-1]-vec_field[:-1,:-1]
  gy = vec_field[:-1,1:]-vec_field[:-1,:-1]

  return np.sqrt(np.linalg.norm(gx,axis=-1)**2+np.linalg.norm(gy,axis=-1)**2)


# Debug
# import pdb; pdb.set_trace()

plt.figure(figsize=(12,8))

plt.subplot(2,4,1)
plt.imshow(e0, cmap='gray')

plt.subplot(2,4,2)
plt.imshow(h0)

plt.subplot(2,4,3)
plt.imshow(gradient_norm(v100), cmap='gray')

plt.subplot(2,4,4)
plt.imshow(v100c)

plt.subplot(2,4,5)
plt.imshow(de100, cmap='gray')

plt.subplot(2,4,6)
gn = gradient_norm(d100)
plt.imshow(gn, cmap='gray')

plt.subplot(2,4,7)
plt.imshow(d100c)

plt.subplot(2,4,8)
#plt.imshow(gn>thresh, cmap='gray')


labels = measure.label(gn>thresh, background=1)
#import pdb; pdb.set_trace()
plt.imshow(labels)


plt.show()
