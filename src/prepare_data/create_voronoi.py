"""Creates Voronoi dataset
Code copied from https://rosettacode.org/wiki/Voronoi_diagram#Python
"""
from PIL import Image
import random
import math
import os
import numpy as np
import h5py

def generate_voronoi_diagram(width, height, num_cells):
    image = Image.new("I", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = []
    ny = []
    idx = []
    for i in range(num_cells):
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        idx.append(i)
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), idx[j])

    return np.array(image)

def gradient_norm(vec_field):
    """Returns norm of gradient at each point in vec_field"""
    gx = vec_field[1:,:-1]-vec_field[:-1,:-1]
    gy = vec_field[:-1,1:]-vec_field[:-1,:-1]
    return np.sqrt(gx**2+gy**2)

def diagram_to_boundary_map(vd):
    """Returns boundary map of voronoi diagram"""
    gn = gradient_norm(vd)
    bound_map = np.zeros_like(vd, dtype=np.float)
    x,y=vd.shape
    bound_map[:x-1,:y-1] += gn
    bound_map[:x-1,1:y] += gn
    bound_map[1:x,:y-1] += gn
    bound_map[1:x,1:y] += gn

    bound_map = bound_map > 0.0
    
    return bound_map
    
#generate_voronoi_diagram(500, 500, 25)

def create_dataset(x,y,z,n_cells):
    """Creates dataset of Voronoi diagrams
    x: width
    y: height
    z: num images
    n_cells: num cells per xy slice
    """
    data_dir = '/usr/people/kluther/Projects/metric_segmentation/data'
    boundaries = np.zeros((z,x,y))
    segs = np.zeros((z,x,y))
    for i in range(z):
        print('Preparing image {}'.format(i+1))
        vd = generate_voronoi_diagram(x,y,n_cells)
        bm = diagram_to_boundary_map(vd)

        boundaries[i] = bm
        segs[i] = vd

    boundary_h5 = h5py.File(os.path.join(data_dir, 'voronoi_boundary.hdf5'), 'w')
    seg_h5 = h5py.File(os.path.join(data_dir, 'voronoi_segmentation.hdf5'), 'w')
    boundary_h5.create_dataset('main', data=boundaries)
    seg_h5.create_dataset('main', data=segs)
    boundary_h5.close()
    seg_h5.close()


create_dataset(1024,1024,32,230)
