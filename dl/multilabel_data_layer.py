# imports
import caffe

import numpy as np
from random import shuffle


class MultilabelDataLayer(caffe.Layer):

	"""
	This is a simple syncronous datalayer for training a multilabel model on
	PASCAL.
	"""

	def setup(self, bottom, top):

		self.top_names = ['data', 'label']

		# === Read input parameters ===

		# params is a python dictionary with layer parameters.
		params = eval(self.param_str)

		# Check the paramameters for validity.
		check_params(params)

		# store input as class variables
		self.batch_size = params['batch_size']

		# Create a batch loader to load the images.
		self.batch_loader = BatchLoader(params, None)

		# === reshape tops ===
		# since we use a fixed input image size, we can shape the data layer
		# once. Else, we'd have to do it in the reshape call.
		top[0].reshape(
			self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
		# Note the 20 channels (because PASCAL has 20 classes.)
		top[1].reshape(self.batch_size, params['n_bits'])

		print_info("MultilabelDataLayer", params)

	def forward(self, bottom, top):
		"""
		Load data.
		"""
		for itt in range(self.batch_size):
			# Use the batch loader to load the next image.
			im, multilabel = self.batch_loader.load_next_image()

			# Add directly to the caffe data layer
			top[0].data[itt, ...] = im
			top[1].data[itt, ...] = multilabel

	def reshape(self, bottom, top):
		"""
		There is no need to reshape the data, since the input is of fixed size
		(rows and columns)
		"""
		pass

	def backward(self, top, propagate_down, bottom):
		"""
		These layers does not back propagate
		"""
		pass


class BatchLoader(object):

	"""
	This class abstracts away the loading of images.
	Images can either be loaded singly, or in a batch. The latter is used for
	the asyncronous data layer to preload batches while other processing is
	performed.
	"""

	def __init__(self, params, result):
		self.result = result
		self.batch_size = params['batch_size']
		self.data = np.load('')
		self.im_shape = params['im_shape']
		self.n_bit = params['n_bit']
		# get list of image indexes.
		self._cur = 0  # current image
		self.indexlist = np.arange(len(self.data[0]), dtype=np.int32)
		# preprocess
		self.img_mean = np.array([128,128,128])

	def load_next_image(self):
		"""
		Load the next image in a batch.
		"""
		# Did we finish an epoch?
		if self._cur == len(self.indexlist):
			self._cur = 0
			shuffle(self.indexlist)

		# Load an image
		index = self.indexlist[self._cur]  # Get the image index
		im = self.data[0][index].astype(np.float32)
		# do a simple horizontal flip as data augmentation
		flip = np.random.choice(2)*2-1
		im = im[:, ::flip, :]
		im = preprocess(im, self.img_mean)

		# Load and prepare ground truth
		multilabel = self.data[1][index]

		self._cur += 1
		return self.transformer.preprocess(im), multilabel


def preprocess(im, img_mean):
	im = np.float32(im)
	im = im[:,:,::-1]
	im -= img_mean
	im = im.transpose((2,0,1))
	return im


def check_params(params):
	"""
	A utility function to check the parameters for the data layers.
	"""
	assert 'split' in params.keys(
	), 'Params must include split (train, val, or test).'

	required = ['batch_size', 'pascal_root', 'im_shape']
	for r in required:
		assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
	"""
	Ouput some info regarding the class
	"""
	print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
		name,
		params['split'],
		params['batch_size'],
		params['im_shape'])
