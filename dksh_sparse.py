import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
from ksh import RBF
import time
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix


class Sparse_DKSH(object):
	def __init__(self, r, m, numlabel, kernel):
		self.r = r # num of hash bits
		self.m = m # num of anchors
		self.kernel = kernel # kernel function
		self.anchors = None # anchor points
		self.W = None # parameter to optimize
		self.numlabel = numlabel
		self.mvec = None # mean vector

		# Hash code and out-of-sample labels
		self.H = None
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

		# pairwise label matrix, S = 2*P*P.T-1_{n*n}
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			P = csc_matrix(trainlabel, dtype=np.float32)
			P = P.T
		else:
			P = csc_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,self.numlabel), dtype=np.float32)
			P = P.T
		H = np.zeros((n,self.r))

		# projection optimization
		RM = np.dot(KK.T, KK)
		W = np.zeros((self.m, self.r), dtype=np.float32) # parameter W
		LM = self.r*(2*np.dot(P.dot(KK).T, P.dot(KK)) - np.dot(np.sum(KK.T, axis=1, keepdims=True), np.sum(KK, axis=0, keepdims=True)))

		# Evaluator
		evaor = ObjEvaluate(0)

		# step 1: initialize with spectral relaxation
		# step 1.1: batch coordinate optimization
		h0 = np.zeros(n)
		print '\nSTEP 1: Initialize with spectral relaxation...'
		print 'step 1.1...'
		for rr in range(self.r):
			if rr > 0:
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)
			(V, U) = eigh(LM, RM, eigvals_only=False)
			W[:,rr] = U[:,self.m-1]
			tmp = np.dot(np.dot(W[:,rr].T, RM), W[:,rr])
			W[:,rr] *= np.sqrt(n/tmp)

			h0 = np.where(np.dot(KK, W[:,rr]) >= 0, 1, -1)
			H[:,rr] = h0
		evaor(P, H, W, KK, mu, self.lmda, self.r, 1)

		# step 1.2: batch coordinate optimization for some loops
		for t in range(5):
			print 'step 1.{}...'.format(t+2)
			for rr in range(self.r):
				h0[:] = H[:,rr]
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM += np.dot(tmp, tmp.T)

				(V, U) = eigh(LM, RM, eigvals_only=False)
				W[:,rr] = U[:,self.m-1]
				tmp = np.dot(np.dot(W[:,rr].T, RM), W[:,rr])
				W[:,rr] *= np.sqrt(n/tmp)

				h0[:] = np.where(np.dot(KK, W[:,rr]) > 0, 1, -1)

				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)

				H[:,rr] = h0

				# PH[:,l+rr] = h0
				# QH[:,l+rr] = -1 * h0
		evaor(P, H, W, KK, mu, self.lmda, self.r, 1)
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
			W = np.dot(invRM, np.dot(KK.T, H))
			# W = -np.dot(invRM, np.dot(KK.T, QH[:,l:]))
			evaor(P, H, W, KK, mu, self.lmda, self.r, 2)

			# step 2.2: fix W, optimize H
			KK_W = np.dot(KK, W)
			for rr in range(self.r):
				if (rr+1) % 10 == 0:
					print 'rr:', rr
				h[:] = H[:,rr]
				H[:,rr] = 0
				fun = lambda x: -self.r*(2*np.sum(P.dot(x)**2) - np.sum(x)**2) + np.sum(np.dot(x,H)**2) - mu * np.dot(KK_W[:,rr], x)
				gra = lambda x: -2*self.r*(2*P.T.dot(P.dot(x)) - np.sum(x)) + 2*np.dot(H,np.dot(x,H)) - mu * KK_W[:,rr]
				res = minimize(fun, h, method='L-BFGS-B', jac=gra, bounds=bnds, options={'disp': False, 'maxiter':500, 'maxfun':500})
				h[:] = np.where(res.x>=0, 1, -1)
				H[:,rr] = h

			evaor(P, H, W, KK, mu, self.lmda, self.r, 1)

		self.W = W
		self.trainlabel = trainlabel
		self.H = np.copy(H)

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

	def __call__(self, P, H, W, KK, mu, lamb, r, flag=0):
		if self.flag == 0 or flag == 0: # do not compute
			return
		elif flag == 1: # update all
			Tmp1 = (2*r)*P.T.dot(P).toarray()-r*np.ones((len(H),len(H))) - np.dot(H,H.T)
			self.Q1 = np.sum(np.sum(Tmp1*Tmp1,axis=1)) / (r**2)
			Tmp2 = np.dot(KK, W) - H
			self.Q2 = (np.sum(np.sum(Tmp2*Tmp2,axis=1)) + lamb*np.sum(np.sum(W*W,axis=1)))
		else:
			Tmp2 = np.dot(KK, W) - H
			self.Q2 = (np.sum(np.sum(Tmp2*Tmp2,axis=1)) + lamb*np.sum(np.sum(W*W,axis=1)))
		print 'Obj Value: Q1={0}, Q2={1}, Total={2}'.format(self.Q1, self.Q2, self.Q1+mu*self.Q2)


def test():
	np.random.seed(47)
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')

	traindata = X[:5000]
	trainlabel = Y[:5000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# train model
	dksh = Sparse_DKSH(32, 1000, 10, RBF)
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
