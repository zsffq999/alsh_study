import numpy as np
from data import hash_value, hash_evaluation
import time
from ksh import RBF


class ITQ(object):
	def __init__(self, r, type, l=0, n_iter=50, reg=1e-4):
		self.r = r
		self.type = type # 'pca', 'cca', 'kernel'
		self.n_iter = n_iter
		self.reg = reg
		self.l = l
		self.W = None

	def train(self, traindata, trainlabel=None):
		n = len(traindata)
		# kernel transformation first
		if self.type == 'kernel':
			pass

		# execute PCA first
		self.mean = np.mean(traindata, axis=0).reshape((1,traindata.shape[1]))
		S = np.cov(traindata.T)
		self.Evecs, _, _ = np.linalg.svd(S)
		self.Evecs = self.Evecs[:,:self.r]

		# encode traindata
		V = np.dot(traindata-self.mean, self.Evecs)

		# PCA-ITQ
		if self.type == 'pca':
			R = np.random.normal(size=(self.r, self.r))
			R, _, _ = np.linalg.svd(R)
			for i in xrange(self.n_iter):
				Z = np.dot(V, R)
				UX = np.where(Z>=0, 1, -1)
				C = np.dot(UX.T, V)
				UB, _, UA = np.linalg.svd(C)
				R = np.dot(UA, UB.T)
			self.W = R
		elif self.type == 'cca' or self.type == 'kernel':
			# make label vector
			Y = np.zeros((n,self.l), dtype=np.float32)
			Y[np.arange(n, dtype=np.int32), trainlabel] = 1
			z = np.hstack((traindata, Y))
			C = np.cov(z.T)

			sx = traindata.shape[1]
			sy = self.l
			Cxx = C[:sx,:sx] + self.reg*np.eye(sx)
			Cxy = C[:sx,sx:]
			Cyx = Cxy.T
			Cyy = C[sx:,sx:] + self.reg*np.eye(sy)

			Rx = np.linalg.cholesky(Cxx)
			invRx = np.linalg.inv(Rx)
			Z = np.dot(np.dot(np.dot(invRx.T, Cxy), np.linalg.solve(Cyy, Cyx)), invRx)
			Z = 0.5*(Z+Z.T)

			r, Wx = np.linalg.eig(Z)   # basis in h (X)
			r = np.sqrt(np.real(r)) # as the original r we get is lamda^2
			Wx = np.dot(invRx, Wx)   # actual Wx values

			index = np.argsort(r)[::-1]
			Wx = Wx[:,index]
			r = r[index]
			print Wx.shape
			self.W = Wx * r.reshape((1,self.r))

	def queryhash(self, qdata):
		if self.type != 'kernel':
			Kdata = np.dot(qdata-self.mean, self.Evecs)
			Y = np.dot(Kdata, self.W)
			Y = np.where(Y>=0, 1, 0)
			return hash_value(Y)
		else:
			pass

	def basehash(self, data):
		return self.queryhash(data)


def test():
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')
	idx = np.arange(60000, dtype=np.int32)
	np.random.shuffle(idx)
	X = X[idx]
	Y = Y[idx]
	traindata = X[:59000]
	trainlabel = Y[:59000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	# train model
	ksh = ITQ(48, 'pca', 10)
	tic = time.clock()
	ksh.train(traindata, trainlabel)
	# ksh = ksh2.KSH(5000, 300, 12, traindata, trainlabel, RBF)
	toc = time.clock()
	print 'time:', toc-tic
	H_base = ksh.basehash(basedata)
	H_test = ksh.queryhash(testdata)

	# evaluate
	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']

if __name__ == "__main__":
	test()