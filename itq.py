import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
import time


class ITQ(object):
	def __init__(self, r, type, l=0, n_iter=50, reg=1e-4):
		self.r = r
		self.type = type # 'pca', 'cca', 'kernel'
		self.n_iter = n_iter
		self.reg = reg
		self.l = l

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
		if type == 'pca':
			R = np.random.normal(self.r, self.r)
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
			Wx = invRx * Wx   # actual Wx values

			index = np.argsort(r)[::-1]
			Wx = Wx[:,index]
			r = r[index]
			self.W = Wx * r.reshape((1,self.r))