import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io
import time
from scipy.sparse.linalg import svds
from data import hash_value, hash_evaluation, sqdist

class DGH(object):
	def __init__(self, r, m, nnanchors=10, sigma=None):
		self.W = None
		self.numhashbits = r
		self.r = r
		self.m = m
		if self.numhashbits >= m:
			valerr = 'The number of hash bits (%s) must be less than the number of anchors (%s).' % (self.numhashbits, m)
			raise ValueError(valerr)
		self.nnanchors = nnanchors
		self.sigma = sigma
		self.n_iter = 5
		self.rou = 5

	def train(self, traindata, trainlabel=None):
		def obj_value(flag=1):
			if flag == 1:
				obj_B = np.trace(np.dot(Z.T.dot(B).T, np.multiply(Lamb.T, (Z.T.dot(B)))))
				obj_Y = np.sum(np.multiply(B,Y))
				print 'obj_B:', obj_B, ', obj_Y:', obj_Y, 'total:', obj_B+self.rou*obj_Y
		n = len(traindata)
		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]

		# determine anchors
		anchoridx = np.copy(indexes)
		np.random.shuffle(anchoridx)
		anchoridx = anchoridx[:self.m]
		self.anchors = traindata[anchoridx]

		(Z, sigma) = self._Z(traindata, self.anchors, self.nnanchors, self.sigma)
		Lamb = 1.0 / Z.sum(axis=0).reshape((1,self.m))

		# Number of iterations: T_R=100, T_B=300, T_G=20
		Tr = 100
		Tb = 300
		Tg = 0
		
		# init: DGH-R
		# Compute A = P \Phi P.T
		# A = (Z \Lamb^-0.5)(Z \Lamb^-0.5).T
		# Computing A equals execute SVD on (Z \Lamb^-0.5)
		# DGH-I
		ZL = scipy.sparse.csr_matrix(Z.multiply(np.sqrt(Lamb)))
		U, s, _ = svds(ZL, self.r+1)
		H = U[:,1:]
		theta = (s*s)[1:].reshape((1,self.r))

		B = np.where(H>=0, 1, -1)
		
		R = np.eye(self.r)
		
		
		# DGH-R
		for _iter in xrange(Tr):
			_U, _, _VT = scipy.linalg.svd(np.dot((H*theta).T, B))
			R = np.dot(_U, _VT)
			B = np.where(np.dot(H*theta, R)>=0, 1, -1)
		
		Y = np.sqrt(n) * np.dot(H, R)
		
		obj_value()
		
		# DGH algorithm
		for _iter in xrange(Tg):
			print 'iter', _iter, '...'
			# B-subproblem
			for i in xrange(Tb):
				Tmp = 2*Z.dot(np.multiply(Lamb.T, (Z.T.dot(B)))) + self.rou*Y
				Tmp = np.where(np.abs(Tmp)<=1e-4, B, Tmp)
				tB = np.where(Tmp>=0, 1, -1)
				if np.max(np.abs(tB-B)) < 1e-4:
					break
				B = tB
			obj_value()
			# Y-subproblem
			B1 = np.sum(B.T, axis=1, keepdims=True)
			
			JB = B - 1.0 / n * np.dot(np.ones((n,1)), np.sum(B, axis=0, keepdims=True))
			U, Sigma, VT = scipy.linalg.svd(JB, full_matrices=False, overwrite_a=True)
			V = VT.T
			ind = self.r - np.searchsorted(Sigma[::-1], 1e-4)
			Y = np.dot(U, V.T) * np.sqrt(n)
			
			'''
			BJB = np.dot(B.T,B) - 1.0/n * np.dot(B1, B1.T)
			V, Sigma2, _ = scipy.linalg.svd(BJB)
			ind = self.r - np.searchsorted(Sigma2[::-1], 1e-5)
			print ind, Sigma2
			inv_sigma = np.sqrt((1.0 / Sigma2[:ind]).reshape((1,ind)))
			BVSigma = np.dot(B, V[:,:ind]*inv_sigma)
			U = np.zeros((n,self.r))
			U[:,:ind] = BVSigma - np.dot(np.ones((n,1)), np.sum(BVSigma, axis=0, keepdims=True))
			print np.dot(U[:,:ind].T, U[:,:ind])
			U[:,ind:] = np.random.rand(n,self.r-ind)
			U[:,ind:] = scipy.linalg.qr(U, mode='economic')[0][:,ind:]
			Y = np.dot(U, V.T) * np.sqrt(n)
			'''
			
			obj_value()

		self.W = np.multiply((Z.T.dot(B)).T, Lamb).T
		self.B = B

	def queryhash(self, data):
		(Z, _) = self._Z(data, self.anchors, self.nnanchors, self.sigma)
		print Z.shape
		Y = Z.dot(self.W)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.B

	@staticmethod
	def _Z(data, anchors, nnanchors, sigma):
		n = data.shape[0]
		m = anchors.shape[0]

		_sqdist = sqdist(data, anchors)
		val = np.zeros((n, nnanchors))
		pos = np.zeros((n, nnanchors), dtype=np.int)
		for i in range(nnanchors):
			pos[:,i] = np.argmin(_sqdist, 1)
			val[:,i] = _sqdist[np.arange(len(_sqdist)), pos[:,i]]
			_sqdist[np.arange(n), pos[:,i]] = float('inf')

		# would be cleaner to calculate sigma in its own separate method, but this is more efficient
		if sigma is None:
			dist = np.sqrt(val[:,nnanchors-1])
			sigma = np.mean(dist) / np.sqrt(2)

		# Next, calculate formula (2) from the paper
		# this calculation differs from the matlab. In the matlab, the RBF kernel's exponent only
		# has sigma^2 in the denominator. Here, 2 * sigma^2. This is accounted for when auto-calculating sigma
		# above by dividing by sqrt(2)

		# Here is how you first calculated formula (2), which is similar in approach to the matlab code (not with
		# respect to the difference mentioned above, though). However, you encountered floating point issues.
		#val = np.exp(-val / (2 * np.power(sigma,2)))
		#s = val.sum(1)[np.newaxis].T # had to do np.newaxis and transpose to make it a column vector
		#														 # just calling ".T" without that wasn't working. reshape would
		#														 # also work. I'm not sure which is preferred.
		#repmat = np.tile(s, (1,nnanchors))
		#val = val / repmat

		# So work in log space and then exponentiate, to avoid the floating point issues.
		# for the denominator, the following code avoids even more precision issues, by relying
		# on the fact that the log of the sum of exponentials, equals some constant plus the log of sum
		# of exponentials of numbers subtracted by the constant:
		#	log(sum_i(exp(x_i))) = m + log(sum_i(exp(x_i-m)))

		c = 2 * np.power(sigma,2) # bandwidth parameter
		exponent = -val / c			 # exponent of RBF kernel
		# no longer using the np.newaxis approach, since you now see keepdims option
		#shift = np.amin(exponent, 1)[np.newaxis].T # np.axis to make column vector
		shift = np.amin(exponent, 1, keepdims=True)
		# no longer using the np.tile approach, since numpy figures it out. You were originally doing
		# exponent - shiftrep, but exponent - shift works the same.
		#shiftrep = np.tile(shift, (1,nnanchors))
		denom = np.log(np.sum(np.exp(exponent - shift), 1, keepdims=True)) + shift
		val = np.exp(exponent - denom)

		Z = scipy.sparse.lil_matrix((n,m))
		for i in range(nnanchors):
			Z[np.arange(n), pos[:,i]] = val[:,i]
		Z = scipy.sparse.csr_matrix(Z)

		return (Z, sigma)


if __name__ == '__main__':

	np.random.seed(97)
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')

	traindata = X[:59000]
	trainlabel = Y[:59000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# train model
	dksh = DGH(48, 300)
	tic = time.clock()
	dksh.train(traindata)
	toc = time.clock()
	print 'time:', toc-tic

	H_test = dksh.queryhash(testdata)
	H_base = dksh.queryhash(basedata)
	
	idx = np.argsort(baselabel)
	print H_base[idx[0:59000:3000]]

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']