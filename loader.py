import numpy as np
import scipy.io as sio


class Loader(object):
	def __init__(self):
		print 'loading data...'

	def split(self, seed):
		pass


class Cifar10Loader(Loader):
	def __init__(self):
		super(Cifar10Loader, self).__init__()
		self.X = np.load('data/cifar_gist.npy')
		self.Y = np.load('data/cifar_label.npy')

	def split(self, seed):
		np.random.seed(seed)
		idx = np.arange(len(self.X), dtype=np.int32)
		np.random.shuffle(idx)

		traindata = self.X[idx[:5000]]
		trainlabel = self.Y[idx[:5000]]
		basedata = self.X[idx[:59000]]
		baselabel = self.Y[idx[:59000]]
		testdata = self.X[idx[59000:]]
		testlabel = self.Y[idx[59000:]]

		return traindata, trainlabel, basedata, baselabel, testdata, testlabel


class Cifar100Loader(Loader):
	def __init__(self):
		super(Cifar100Loader, self).__init__()
		self.train = sio.loadmat('data/cifar_100/train_gist.mat')
		self.test = sio.loadmat('data/cifar_100/test_gist.mat')
		self.n = len(self.train['gist'])

	def split(self, seed):
		np.random.seed(seed)
		idx = np.arange(self.n, dtype=np.int32)
		np.random.shuffle(idx)

		traindata = self.train['gist'][idx[:5000]]
		trainlabel = self.train['fine_labels'][idx[:5000],0]
		basedata = self.train['gist']
		baselabel = self.train['fine_labels'][:,0]
		testdata = self.test['gist'][:1000]
		testlabel = self.test['fine_labels'][:1000,0]

		return traindata, trainlabel, basedata, baselabel, testdata, testlabel