import numpy as np
from scipy.linalg import eigh
import time
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix
import sys
import cPickle as cp

sys.path.extend(['../','../../VGG_ILSVRC2012/fc7_npydata/'])

root = '../../VGG_ILSVRC2012/fc7_npydata/'

from data import hash_value, hash_evaluation


class DKSH_ImageNet(object):
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

	def train(self):
		trainlabel = np.load(root+'trainlabel.npy')
		n = len(trainlabel)
		self.anchors = np.load('anchors.npy')
		mu = self.mu * n

		KK = np.zeros((n, self.m))
		pt = 0
		print 'making kernel vector...'
		for i in xrange(13):
			print 'loading', i+1, '...'
			X = np.load(root+'traindata_fc7_norm_{}.npy'.format(i+1))
			t = X.shape[0]
			KK[pt:pt+t,:] = self.kernel(X, self.anchors)
			pt += t
			del X

		# kernel matrix and mean
		self.mvec = np.mean(KK, axis=0).reshape((1, self.m))
		KK = KK - self.mvec

		np.save('mean_vec.npy', self.mvec)

		# pairwise label matrix, S = 2*P*P.T-1_{n*n}
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			P = csc_matrix(trainlabel, dtype=np.float32)
			P = P.T
		else:
			P = csc_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,self.numlabel), dtype=np.float32)
			P = P.T
		H = np.zeros((n,self.r))

		print 'preparing step 1...'

		# projection optimization
		RM = np.dot(KK.T, KK)
		W = np.zeros((self.m, self.r), dtype=np.float32) # parameter W
		LM = self.r*(2*np.dot(P.dot(KK).T, P.dot(KK)) - np.dot(np.sum(KK.T, axis=1, keepdims=True), np.sum(KK, axis=0, keepdims=True)))

		# Evaluator

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

		# step 1.2: batch coordinate optimization for some loops
		h1 = np.zeros(n)
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

				# JUST FOR EXPERIMENT
				h1[:] = np.where(np.dot(KK, W[:,rr]) > 0, 1, -1)
				H[:,rr] = 0
				fun = lambda x: -self.r*(2*np.sum(P.dot(x)**2) - np.sum(x)**2) + np.sum(np.dot(x,H)**2)
				if fun(h1) <= fun(h0):
					h0[:] = h1[:]
				# END FOR EXPERIMENT

				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)

				H[:,rr] = h0

				np.save('hash_value.npy', hash_value(np.where(H>=0, 1, 0)))
				np.save('weight.npy', W)

		# step 2: discrete optimization
		print '\nSTEP 2: Discrete Optimization...'
		RM += self.lmda * np.eye(self.m)
		invRM = np.linalg.inv(RM)
		h = np.zeros(n)
		h1 = np.zeros(n)
		bnds = [(-1,1) for i in xrange(n)]
		for t in range(5):
			print '\nIter No: %d' % t
			# step 2.1: fix Hp, Hq, optimize W
			W = np.dot(invRM, np.dot(KK.T, H))

			# step 2.2: fix W, optimize H
			KK_W = np.dot(KK, W)
			np.save('hash_value.npy', hash_value(np.where(KK_W>=0, 1, 0)))
			np.save('weight.npy', W)
			for rr in range(self.r):
				if (rr+1) % 8 == 0:
					print 'rr:', rr
				h[:] = H[:,rr]
				H[:,rr] = 0
				fun = lambda x: -self.r*(2*np.sum(P.dot(x)**2) - np.sum(x)**2) + np.sum(np.dot(x,H)**2) - mu * np.dot(KK_W[:,rr], x)
				gra = lambda x: -2*self.r*(2*P.T.dot(P.dot(x)) - np.sum(x)) + 2*np.dot(H,np.dot(x,H)) - mu * KK_W[:,rr]
				res = minimize(fun, h, method='L-BFGS-B', jac=gra, bounds=bnds, options={'disp': False, 'maxiter':500, 'maxfun':500})
				h1[:] = np.where(res.x>=0, 1, -1)
				# JUST FOR EXPERIMENT
				if fun(h1) <= fun(h):
					H[:,rr] = h1
				else:
					H[:,rr] = h
				# END FOR EXPERIMENT

		self.W = W
		self.trainlabel = trainlabel

		Y = np.dot(KK, self.W)
		Y = np.where(Y>=0, 1, 0)
		np.save('base_hash_value.npy', hash_value(Y))
		np.save('weight.npy', self.W)


	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.anchors)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.W)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)


def RBF(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.exp(2*np.dot(X,Y.T) - X2 - Y2)


def make_anchors():
	anchors = np.zeros((1000, 4096))
	count = np.zeros(1000)
	Xlabel = np.load(root+'trainlabel.npy')
	pt = 0
	print 'step 1...'
	for i in xrange(13):
		print 'loading', i+1, '...'
		X = np.load(root+'traindata_fc7_norm_{}.npy'.format(i+1))
		n = X.shape[0]
		for (j,x) in enumerate(X):
			label = Xlabel[pt+j]
			count[label] += 1
			anchors[label] += x
		pt += n
		del X
	# normalize
	anchors /= np.linalg.norm(anchors, axis=1).reshape((1000,1))
	np.save('anchors.npy', anchors)

	res_anchor = np.zeros((1000, 4096))
	inner_prod = np.zeros(1000)
	print 'step 2...'
	pt = 0
	for i in xrange(13):
		print 'loading', i+1, '...'
		X = np.load(root+'traindata_fc7_norm_{}.npy'.format(i+1))
		n = X.shape[0]
		for (j,x) in enumerate(X):
			label = Xlabel[pt+j]
			ip = np.dot(anchors[label], x)
			if inner_prod[label] < ip:
				res_anchor[label] = x
				inner_prod[label] = ip
		pt += n
		del X

	np.save('anchors.npy', res_anchor)


if __name__ == "__main__":
	tic = time.clock()
	alg = DKSH_ImageNet(128, 1000, 1000, RBF)
	alg.train()
	toc = time.clock()
	print 'time:', toc-tic

	with open('DKSH_ImageNet.alg', 'wb') as f:
		cp.dump(alg, f)
	print 'training imagenet finished!'

	print 'hashing test vector...'
	H_test = alg.queryhash(np.load(root+'testdata_fc7_norm.npy'))
	np.save('query_hash_value.npy')
	print 'Done.'
