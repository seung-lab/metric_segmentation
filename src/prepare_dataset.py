"""Preprocesses dataset
1. Splits objects if they are not touching a z-slice
"""
import numpy as np
import h5py
from skimage import measure
import os

# Load original dataset
data_dir = '/usr/people/kluther/Projects/metric_segmentation/data/'
seg_h5_original = h5py.File(os.path.join(data_dir,'human_labels.h5'),'r')
seg_data_original = seg_h5_original['main']

# Split objects that are not connected
seg_data_split = np.zeros_like(seg_data_original, dtype=int)
for i, slc in enumerate(seg_data_original):
    if (i+1) % 5 == 0: print(i+1);
    slc_split = np.zeros_like(slc, dtype=int)
    #import pdb; pdb.set_trace()
    for seg_id in np.unique(slc):
        if seg_id != 0:
            split_ids = measure.label(slc == seg_id, background=0)
            slc_split[slc==seg_id] = split_ids[slc==seg_id] + seg_id*2**10

    seg_data_split[i] = slc_split


# Save split segmentation 
seg_h5 = h5py.File(os.path.join(data_dir, "human_labels_split.h5"),'w')
seg_h5.create_dataset('main', data=seg_data_split)
seg_h5.close()
