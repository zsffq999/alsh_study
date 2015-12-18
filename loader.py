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


class NuswideLoader(Loader):
	def __init__(self):
		self.data = np.vstack((np.load('data/nuswide/bow_train.npy'), np.load('data/nuswide/bow_test.npy')))
		self.data /= np.linalg.norm(self.data, axis=1).reshape((1,self.data.shape[1]))
		self.label = np.vstack((np.load('data/nuswide/label_train.npy'), np.load('data/nuswide/label_test.npy')))


	def split(self, seed):
		np.random.seed(seed)
		n = self.data.shape[0]
		idx = np.arange(n, dtype=np.int32)
		np.random.shuffle(idx)

		label_cnt = 500 * np.ones(21, dtype=np.int32)
		train_sample = []
		for i in idx:
			tmp = np.argwhere(self.label[i]==1)[:,0]
			for l in tmp:
				if label_cnt[l] > 0:
					train_sample.append(i)
					label_cnt[l] -= 1
			if len(train_sample) == 10500:
				break
		train_sample = np.array(train_sample, dtype=np.int32)

		label_cnt = 100 * np.ones(21, dtype=np.int32)
		test_sample = []
		for i in idx[::-1]:
			tmp = np.argwhere(self.label[i]==1)[:,0]
			for l in tmp:
				if label_cnt[l] > 0:
					test_sample.append(i)
					label_cnt[l] -= 1
			if len(test_sample) == 2100:
				break
		test_sample = np.array(test_sample, dtype=np.int32)
		base_sample = np.setdiff1d(idx, test_sample)
		return self.data[train_sample], self.label[train_sample], self.data[base_sample], self.label[base_sample], self.data[test_sample], self.label[test_sample]

