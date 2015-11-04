import numpy as np
from data import hash_value, hash_evaluation
from scipy.linalg import eigh
import time
import ksh2


def OptProjectionFast(K, PH, QH, a0, cn):
	(n, m) = K.shape
	cost = np.zeros(cn+2, dtype=np.float32)
	y = np.dot(K, a0)
	y = 2.0 / (1+np.exp(-1*y)) - 1
	cost[0] = -np.dot(np.dot(y.T,PH), np.dot(QH.T,y))
	# cost[0] = -np.dot(np.dot(y.T,S), y)
	cost[1] = cost[0]

	a1 = a0 
	delta = np.zeros(cn+2)
	delta[0] = 0
	delta[1] = 1

	beta = np.zeros(cn+1)
	beta[0] = 1

	for t in range(0, cn):
		alpha = (delta[t]-1)/delta[t+1]
		v = a1 + alpha*(a1-a0)
		y0 = np.dot(K, v)
		y1 = 2.0 / (1+np.exp(-1*y0)) - 1
		gv = -np.dot(np.dot(y1.T,PH), np.dot(QH.T,y1))
		# gv = -np.dot(np.dot(y1.T, S), y1)
		ty = np.dot(PH, np.dot(QH.T,y1)) * (np.ones(n)-y1**2)
		# ty = np.multiply(np.dot(S,y1), (np.ones((n,1))-y1**2))[0]
		dgv = -np.dot(K.T, ty)

		# seek beta
		flag = 0
		for j in range(0,51):
			b = 2**j * beta[t]
			z = v-dgv/b
			y0 = np.dot(K,z)
			y1 = 2.0 / (1+np.exp(-1*y0)) - 1
			gz = -np.dot(np.dot(y1.T,PH), np.dot(QH.T,y1))
			# gz = -np.dot(np.dot(y1.T, S), y1)
			dif = z-v
			gvz = gv + np.dot(dgv.T, dif) + np.dot(b*dif.T, dif)/2

			if gz <= gvz:
				flag = 1
				beta[t+1] = b
				a0 = a1 
				a1 = z
				cost[t+2] = gz
				break

		if flag == 0:
			# t = t-1
			break
		else:
			delta[t+2] = (1+np.sqrt(1+4*delta[t+1]**2))/2

		if abs(cost[t+2] - cost[t+1])/n <= 1e-2:
			break

	a = a1 
	cost = cost/n
	return a, cost


class KSH(object):
	def __init__(self, r, m, numlabel, kernel):
		self.r = r # num of hash bits
		self.m = m # num of anchors
		self.kernel = kernel # kernel function
		self.anchors = None # anchor points
		self.A = None # parameter to optimize
		self.numlabel = numlabel
		self.mvec = None # mean vector

	def train(self, traindata, trainlabel):
		n = len(traindata)
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

		# PH = [P, +H], QH = [Q, -H]
		PH = np.zeros((n,l+self.r), dtype=np.float32)
		PH[np.arange(n, dtype=np.int32), trainlabel] = 1
		QH = np.copy(PH)
		PH[:,:l-1] *= 2*self.r
		PH[:,l-1] = self.r
		QH[:,l-1] = -1
		S = np.dot(PH[:,:l], QH.T[:l]) / self.r
		print 'Init Obj:', np.sum(np.sum(S*S,axis=1)), np.sum(S)

		# projection optimization
		RM = np.dot(KK.T, KK)
		A = np.zeros((self.m, self.r), dtype=np.float32) # parameter A
		LM = np.dot(np.dot(KK.T, PH[:,:l]), np.dot(QH.T[:l], KK))

		# hash matrix
		H = np.zeros((n, self.r), dtype=np.int8)

		# greedy optimization
		for rr in range(0, self.r):
			print "No:", rr

			# step 1: spectral relaxation
			if rr > 0:
				tmp = np.dot(KK.T, h0.reshape((n,1)))
				LM -= np.dot(tmp, tmp.T)
			(V, U) = eigh(LM, RM, eigvals_only=False)
			A[:,rr] = U[:,self.m-1]
			tmp = np.dot(np.dot(A[:,rr].T, RM), A[:,rr])
			A[:,rr] *= np.sqrt(n/tmp)


			# step 2: sigmoid smoothing
			get_vec, cost = OptProjectionFast(KK, PH[:,:l+rr], QH[:,:l+rr], A[:,rr], 500)

			h0 = np.dot(KK, A[:,rr])
			h0 = np.where(h0>=0, 1, -1)

			h1 = np.dot(KK, get_vec)
			h1 = np.where(h1>=0, 1, -1)

			if np.dot(np.dot(PH.T[:l+rr], h1), np.dot(QH.T[:l+rr], h1)) > \
				np.dot(np.dot(PH.T[:l+rr], h0), np.dot(QH.T[:l+rr], h0)):
				A[:,rr] = get_vec
				h0 = h1

			# update PH and QH
			PH[:,l+rr] = h0
			QH[:,l+rr] = -1 * h0
			Tmp = S + np.dot(PH[:,l:l+rr+1], QH[:,l:l+rr+1].T) / (rr+1)
			print "Obj:", np.sum(np.sum(Tmp*Tmp,axis=1))
			H[:,rr] = (h0 > 0).astype(np.int8)

		self.A = A

		return hash_value(H)

	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.anchors)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.A)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)


def test_matrix_fac():
	labels = (np.random.rand(100) * 10).astype(np.int32)
	P = np.zeros((100, 11))
	P[np.arange(100, dtype=np.int32), labels] = 1
	Q = np.copy(P)
	P *= 2
	P[:,-1] = 1
	Q[:,-1] = -1
	S = -np.ones((100, 100))
	for i in xrange(100):
		for j in xrange(100):
			if labels[i] == labels[j]:
				S[i,j] = 1

	print labels
	print P[:10]
	print Q[:10]
	print np.linalg.norm(S-np.dot(P, Q.T))

# test_matrix_fac()


def RBF(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.exp(2*np.dot(X,Y.T) - X2 - Y2)


def test():
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')
	idx = np.arange(60000, dtype=np.int32)
	np.random.shuffle(idx)
	X = X[idx]
	Y = Y[idx]
	traindata = X[:5000]
	trainlabel = Y[:5000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	# train model
	ksh = KSH(48, 300, 10, RBF)
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