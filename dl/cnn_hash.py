from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

import sys
sys.path.extend('./')
import numpy as np
from multilabel_data_layer import *


class CNN_hash(object):
	def __init__(self, n_bit):
		# some configures
		self.solver_proto = "dl/python_cifar10_6conv_solver.prototxt"
		self.device = 0
		self.fine_model_dir = ""
		self.out_blob = "ip2"
		self.snapshot_file = 'cifar10_6conv_iter_0.solverstate'
		self.hash_model_dir = 'dl/hash_model.caffemodel'

		self.n_bit = n_bit

		caffe.set_mode_gpu()
		caffe.set_device(self.device)
		self.solver = caffe.SGDSolver(self.solver_proto)
		for k, v in self.solver.net.blobs.items():
			print k, v.data.shape
		assert self.solver.net.blobs[self.out_blob].data.shape[1] == self.n_bit

		if self.fine_model_dir != "":
			self.solver.net.copy_from(self.fine_model_dir)
		self.solver.snapshot()
		self.solver.net.save(self.hash_model_dir)

	def train(self, traindata, H):
		self.solver.restore(self.snapshot_file)
		self.solver.net.copy_from(self.hash_model_dir)

		self.solver.net.layers[0].batch_loader.update_data(traindata, H)
		self.solver.test_nets[0].layers[0].batch_loader.update_data(traindata, H)

		self.solver.solve()
		self.solver.save(self.hash_model_dir)

	def predict(self, traindata):
		self.solver.net.layers[0].batch_loader.update_data(traindata)
		Y = np.zeros((len(traindata), self.n_bit))

		_cur = 0
		_batch_size = self.solver.net.layers[0].batch_size
		_n = len(traindata)
		while _cur <= _n:
			self.solver.net.forward()
			if _cur+_batch_size <= _n:
				Y[_cur:_cur+_batch_size] = self.solver.net.blobs[self.out_blob].data
			else:
				Y[_cur:] = self.solver.net.blobs[self.out_blob].data[:_n-_cur]
			_cur += _batch_size

		return Y