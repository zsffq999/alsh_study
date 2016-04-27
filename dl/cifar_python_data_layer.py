# imports
import caffe

import numpy as np
from random import shuffle
import cPickle as cp
import scipy.io as sio


class PythonDataLayer(caffe.Layer):

	"""
	This is a simple syncronous datalayer for training a multilabel model on
	CIFAR.
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
		self.phase = params['phase']

		# Create a batch loader to load the images.
		self.batch_loader = BatchLoader(params, None)

		# === reshape tops ===
		# since we use a fixed input image size, we can shape the data layer
		# once. Else, we'd have to do it in the reshape call.
		top[0].reshape(
			self.batch_size, 3, params['height'], params['width'])
		# Note the 20 channels (because PASCAL has 20 classes.)
		top[1].reshape(self.batch_size)

		print_info("PythonDataLayer", params)

	def forward(self, bottom, top):
		"""
		Load data.
		"""
		for itt in range(self.batch_size):
			# Use the batch loader to load the next image.
			im, multilabel = self.batch_loader.load_next_image()

			# Add directly to the caffe data layer
			top[0].data[itt, ...] = im
			top[1].data[itt] = multilabel

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
		self.height = params['height']
		self.width = params['width']
		self.is_train = (params['phase']=='TRAIN')
		# get data
		self.data = (np.load('cifar10_data/cifar10_data.npy'), np.load('cifar10_data/cifar10_label.npy'))
		# get list of image indexes.
		self._cur = 0  # current image
		self.n_data = 50000 if self.is_train else 10000
		self.indexlist = np.arange(self.n_data, dtype=np.int32) if self.is_train else 50000+np.arange(self.n_data, dtype=np.int32)
		# preprocess: compute img mean
		self.img_mean = np.array([120.7,120.7,120.7])

	def load_next_image(self):
		"""
		Load the next image in a batch.
		"""
		# Did we finish an epoch?
		if self._cur == len(self.indexlist):
			self._cur = 0
			if self.is_train:
				shuffle(self.indexlist)

		# Load an image
		index = self.indexlist[self._cur]  # Get the image index
		im = self.data[0][index].astype(np.float32)
		# do a simple horizontal flip as data augmentation
		if self.is_train:
			flip = np.random.choice(2)*2-1
		im = im[:, ::flip, :]
		im = preprocess(im, self.img_mean)

		# Load and prepare ground truth
		multilabel = self.data[1][index]

		self._cur += 1
		return im, multilabel


def preprocess(im, img_mean):
	im = np.float32(im)
	im -= img_mean.reshape((3,1,1))
	im = im[::-1,:,:]
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
