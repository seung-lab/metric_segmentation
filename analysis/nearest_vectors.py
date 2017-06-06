"""Given an object id, return its mean vector and find its nearest neighbors"""
import numpy as np
import h5py

# Get object ids (img + seg id within img) and centroids
# Get mean vector for each object
# Create fn to find nearest object given a vector
# Create fn to display these objects

def find_object_ids_centroids_vectors(id_imgs, vec_imgs):
  """Returns dict (id, centroid) for all objects.
      Note that we treat objects in different slices as different objects
      Note that id = 2**16 * slice_no + original id (called slc_id)
  """
  # Iterate through each object
  centroids = {}
  vectors = {}
  for i in range(len(id_imgs)):
    id_img = id_imgs[i]
    vec_img = vec_imgs[i]

    slc_ids = np.unique(id_img)
    #if i == 20: break
    if i % 10 == 0: print(i)
#    import pdb; pdb.set_trace()
    for slc_id in slc_ids:
      if slc_id != 0:
        obj_id = i*2**15+slc_id
        mask = np.expand_dims(id_img == slc_id, axis=-1)
        centroid = compute_centroid(mask)
        centroid.append(i)
        mean_vector = compute_mean_vector(vec_img, mask)

        centroids[obj_id] = centroid
        vectors[obj_id] = mean_vector

  return centroids, vectors

def compute_centroid(mask):
  """Returns centroid of object scl_id in 2d array scl"""
  return [np.mean(l).astype(np.int32) for l in np.where(mask)][0:2]

def compute_mean_vector(vec_img, mask):
  """Returns mean vector for object specified by mask"""
  return np.sum(vec_img*mask, axis=(0,1))/np.sum(mask)

def affinity(v1,v2):
  return np.exp(-np.linalg.norm(v1-v2))

def k_nearest_neighbors(vectors, obj_id, k):
  """Returns the k objects with closest mean vectors"""
  vec_1 = vectors[obj_id]
  affinities = [(id_2, affinity(vec_1, vec_2)) for id_2, vec_2 in vectors.items() if obj_id != id_2]
  affinities = sorted(affinities, key=lambda x: x[1], reverse=True)

  return affinities[:k]

