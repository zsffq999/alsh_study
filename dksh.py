import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
from ksh import RBF
import time
from scipy.optimize import minimize


class DKSHv2(object):
	def __init__(self, r, m, numlabel, kernel):
		self.r = r # num of hash bits
		self.m = m # num of anchors
		self.kernel = kernel # kernel function
		self.anchors = None # anchor points
		self.W = None # parameter to optimize
		self.numlabel = numlabel
		self.mvec = None # mean vector

		# Hash code and out-of-sample labels
		self.Hp = None # database hash code
		self.Hq = None # query hash code
		self.trainlabel = None

		# tuning parameters
		self.mu = 1e-4
		self.lmda = 0

	def train(self, traindata, trainlabel):
		n = len(traindata)
		mu = self.mu * n
		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]
		trainlabel = trainlabel[indexes]

		# determine anchors
		anchoridx = np.copy(indexes)
		np.random.shuffle(anchoridx)
		anchoridx = anchoridx[:self.m]
		self.anchors = traindata[anchoridx]

		# kernel matrix and mean
		KK = self.kernel(traindata, self.anchors)
		self.mvec = np.mean(KK, axis=0).reshape((1, self.m))
		KK = KK - self.mvec

		# pairwise label matrix, rS=PQ^T
		l = self.numlabel + 1

		# PH = [P, +Hp], QH = [Q, -Hq]
		PH = np.zeros((n,l+self.r), dtype=np.float32)
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			PH[:,:self.numlabel] = trainlabel
		else:
			PH[np.arange(n, dtype=np.int32), trainlabel] = 1
		QH = np.copy(PH)
		PH[:,:l-1] *= 2 * self.r
		PH[:,l-1] = self.r
		QH[:,l-1] = -1

		# projection optimization
		RM = np.dot(KK.T, KK)
		W = np.zeros((self.m, self.r), dtype=np.float32) # parameter W
		LM = np.dot(np.dot(KK.T, PH[:,:l]), np.dot(QH.T[:l], KK))

		# Evaluator
		evaor = ObjEvaluate(0)

		# step 1: initialize with spectral relaxation
		# step 1.1: batch coordinate optimization
		h0 = np.zeros(n)
		print '\nSTEP 1: Initialize with spectral relaxation...'
		for rr in range(self.r):

			if rr > 0:
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)
			(V, U) = eigh(LM, RM, eigvals_only=False)
			W[:,rr] = U[:,self.m-1]
			tmp = np.dot(np.dot(W[:,rr].T, RM), W[:,rr])
			W[:,rr] *= np.sqrt(n/tmp)

			h0 = np.where(np.dot(KK, W[:,rr]) >= 0, 1, -1)
			PH[:,l+rr] = h0
			QH[:,l+rr] = -1 * h0
		evaor(PH, QH, W, KK, mu, self.lmda, self.r, 1)

		# step 1.2: batch coordinate optimization for some loops
		for t in range(5):
			for rr in range(self.r):
				h0[:] = PH[:,l+rr]
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM += np.dot(tmp, tmp.T)

				(V, U) = eigh(LM, RM, eigvals_only=False)
				W[:,rr] = U[:,self.m-1]
				tmp = np.dot(np.dot(W[:,rr].T, RM), W[:,rr])
				W[:,rr] *= np.sqrt(n/tmp)

				h0[:] = np.where(np.dot(KK, W[:,rr]) > 0, 1, -1)

				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)

				PH[:,l+rr] = h0
				QH[:,l+rr] = -1 * h0
		evaor(PH, QH, W, KK, mu, self.lmda, self.r, 1)
		# TSP
		# PH[:,l:] = np.where(np.random.rand(n,self.r)>0.5,1,-1)
		# QH[:,l:] = -PH[:,l:]
		# evaor(PH, QH, W, KK, mu, self.lmda, self.r, 1)

		# step 2: discrete optimization
		print '\nSTEP 2: Discrete Optimization...'
		RM += self.lmda * np.eye(self.m)
		invRM = np.linalg.inv(RM)
		h = np.zeros(n)
		bnds = [(-1,1) for i in xrange(n)]
		for t in range(5):
			print '\nIter No: %d' % t
			# step 2.1: fix Hp, Hq, optimize W
			W = -np.dot(invRM, np.dot(KK.T, QH[:,l:]))
			evaor(PH, QH, W, KK, mu, self.lmda, self.r, 2)

			# step 2.2: fix W, optimize H
			KK_W = np.dot(KK, W)
			for rr in range(self.r):
				h[:] = PH[:,l+rr]
				QH[:,l+rr] = PH[:,l+rr] = 0
				fun = lambda x: -np.dot(np.dot(x,PH), np.dot(x,QH)) - mu * np.dot(KK_W[:,rr], x)
				gra = lambda x: -2*np.dot(PH, np.dot(x, QH)) - mu * KK_W[:,rr]
				res = minimize(fun, h, method='L-BFGS-B', jac=gra, bounds=bnds, options={'disp': False})
				h[:] = np.where(res.x>=0, 1, -1)
				PH[:,l+rr] = h
				QH[:,l+rr] = -1 * h

			evaor(PH, QH, W, KK, mu, self.lmda, self.r, 1)

		self.W = W
		self.trainlabel = trainlabel
		self.Hp = np.copy(PH[:,l:])
		self.Hq = np.copy(-QH[:,l:])

		QH[:,l:] = np.where(np.dot(KK, self.W)>=0, -1, 1)
		PH[:,l:] = -QH[:,l:]
		evaor(PH, QH, W, KK, mu, self.lmda, self.r, 1)

	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.anchors)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.W)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)


class ObjEvaluate(object):
	def __init__(self, flag=0):
		self.Q1 = 0
		self.Q2 = 0
		self.flag = flag

	def __call__(self, PH, QH, W, KK, mu, lamb, r, flag=0):
		if self.flag == 0 or flag == 0: # do not compute
			return
		elif flag == 1: # update all
			Tmp1 = np.dot(PH, QH.T)
			self.Q1 = np.sum(np.sum(Tmp1*Tmp1,axis=1)) / (r**2)
			Tmp2 = np.dot(KK, W) + QH[:,-r:]
			self.Q2 = (np.sum(np.sum(Tmp2*Tmp2,axis=1)) + lamb*np.sum(np.sum(W*W,axis=1)))
		else:
			Tmp2 = np.dot(KK, W) + QH[:,-r:]
			self.Q2 = (np.sum(np.sum(Tmp2*Tmp2,axis=1)) + lamb*np.sum(np.sum(W*W,axis=1)))
		print 'Obj Value: Q1={0}, Q2={1}, Total={2}'.format(self.Q1, self.Q2, self.Q1+mu*self.Q2)


def test():
	np.random.seed(47)
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')

	traindata = X[:59000]
	trainlabel = Y[:59000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# train model
	dksh = DKSHv2(32, 1000, 10, RBF)
	tic = time.clock()
	dksh.train(traindata, trainlabel)
	toc = time.clock()
	print 'time:', toc-tic

	H_test = dksh.queryhash(testdata)
	H_base = dksh.queryhash(basedata)

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']

if __name__ == "__main__":
	test()
