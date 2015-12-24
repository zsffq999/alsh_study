import numpy as np
from data import hash_value


class LSH(object):
	def __init__(self, r):
		self.r = r

	def train(self, traindata, trainlabel=None):
		dim = traindata.shape[1]
		self.W = np.random.normal(size=(dim, self.r))

	def queryhash(self, qdata):
		Y = np.dot(qdata, self.W)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)