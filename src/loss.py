import tensorflow as tf

def static_shape(x):
	tmp=[x.value for x in x.get_shape()]
	return tmp

def bounded_cross_entropy(guess,truth):
	guess = 0.999998*guess + 0.000001
	return  - truth * tf.log(guess) - (1-truth) * tf.log(1-guess)

def label_diff(x,y):
	return tf.to_float(tf.equal(x,y))

def get_pair(A,offset, patch_size):
	os1 = [max(0,x) for x in offset]
	os2 = [max(0,-x) for x in offset]

	A1 = A[:,os1[0]:patch_size[0]-os2[0],
		os1[1]:patch_size[1]-os2[1],
		os1[2]:patch_size[2]-os2[2],
		:]
	A2 = A[:,os2[0]:patch_size[0]-os1[0],
		os2[1]:patch_size[1]-os1[1],
		os2[2]:patch_size[2]-os1[2],
		:]
	return (A1, A2)

def affinity(x, y):
	displacement = x - y
	interaction = tf.reduce_sum(
		displacement * displacement,
		reduction_indices=[4],
		keep_dims=True)
	return tf.exp(-0.5 * interaction)

#We will only count edges with at least one endpoint in mask
def long_range_loss_fun(vec_labels, human_labels, offsets, mask):
	patch_size = static_shape(vec_labels)[1:4]
	cost = 0
	otpts = {}


	for i, offset in enumerate(offsets):
		guess = affinity(
			*get_pair(vec_labels,offset,patch_size))
		truth = label_diff(
				*get_pair(human_labels,offset,patch_size))

		curr_mask = tf.maximum(*get_pair(mask, offset, patch_size))

		otpts[offset] = guess

		cost += tf.reduce_sum(curr_mask * bounded_cross_entropy(guess, truth))

	return cost, otpts
