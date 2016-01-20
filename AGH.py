import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io
import time
from data import hash_value, hash_evaluation, sqdist


class AGH(object):
	def __init__(self, r, m, nnanchors=2, sigma=None):
		self.W = None
		self.numhashbits = r
		self.m = m
		if self.numhashbits >= m:
			valerr = 'The number of hash bits (%s) must be less than the number of anchors (%s).' % (self.numhashbits, m)
			raise ValueError(valerr)
		self.nnanchors = nnanchors
		self.sigma = sigma

	def train(self, traindata):
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
		self.sigma = sigma
		self.W = self._W(Z, self.numhashbits)

	def queryhash(self, data):
		(Z, _) = self._Z(data, self.anchors, self.nnanchors, self.sigma)
		Y = Z.dot(self.W)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)

	@staticmethod
	def _W(Z, numhashbits):
		s = np.asarray(Z.sum(0)).ravel() # extra steps here are for compatibility with sparse matrices
		isrl = np.diag(np.power(s, -0.5)) # isrl = inverse square root of lambda
		ztz = Z.T.dot(Z) # ztz = Z transpose Z
		if scipy.sparse.issparse(ztz):
			ztz = ztz.todense()
		M = np.dot(isrl, np.dot(ztz, isrl))
		eigenvalues, V = scipy.linalg.eig(M) # there is also a numpy.linalg.eig
		I = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[I]
		V = V[:,I]

		# this is also essentially what they do in the matlab since check for equality to 1
		# doesn't work because of floating point precision
		if eigenvalues[0] > 0.99999999:
			eigenvalues = eigenvalues[1:]
			V = V[:,1:]
		eigenvalues = eigenvalues[0:numhashbits]
		V = V[:,0:numhashbits]
		# paper also multiplies by sqrt(n), but their matlab code doesn't. isn't necessary.

		W = np.dot(isrl, np.dot(V, np.diag(np.power(eigenvalues, -0.5))))
		return W

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

	traindata = X[:5000]
	trainlabel = Y[:5000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# train model
	dksh = AGH(32, 300)
	tic = time.clock()
	dksh.train(traindata)
	toc = time.clock()
	print 'time:', toc-tic

	H_test = dksh.queryhash(testdata)
	H_base = dksh.queryhash(basedata)

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']