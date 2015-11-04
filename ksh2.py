from numpy import *
import cPickle as pickle
from scipy.linalg import eigh
from data import *
import random
import time


def sqdist(a,b):
	d  = mat('')
	aa = power(a,2).sum(0).reshape((1, a.shape[1]))
	bb = power(b,2).sum(0).reshape((1, b.shape[1]))

	ab = dot(a.T,b)
	d = repmat( aa.T, 1, bb.shape[1] ) + repmat( bb, aa.shape[0], 1) - 2*ab

	return d


def OptProjectionFast(K, S, a0, cn):
	(n, m) = K.shape
	cost   = zeros( cn+2, dtype=float32 )
	y = dot(K, a0) 
	y = 2*(1+exp(-1*y))**-1 - 1
	cost[0] = -dot(dot(y.T,S), y)
	cost[1] = cost[0]

	a1 = a0 
	delta = zeros(cn+2)
	delta[0] = 0
	delta[1] = 1

	beta = zeros(cn+1)
	beta[0] = 1

	for t in range(0, cn):
		alpha = (delta[t]-1)/delta[t+1]
		v = a1 + alpha*(a1-a0)
		y0 = dot(K, v)
		y1 = 2*(1+exp(-1*y0))**-1 -1
		gv = -dot(dot(y1.T, S), y1)
		ty = multiply( dot(S,y1), (ones((n,1))-y1**2) )[0] 
		dgv = -dot( K.T, ty)

		# seek beta
		flag = 0
		for j in range(0,51):
			b  = 2**j * beta[t]
			z  = v-dgv/b
			y0 = dot(K,z)
			y1 = 2*(1+exp(-1*y0))**-1 - 1
			gz = -dot(dot(y1.T, S), y1)
			dif= z-v 
			gvz= gv + dot(dgv.T, dif) + dot(b*dif.T, dif)/2

			if gz <= gvz:
				flag = 1
				beta[t+1] = b
				a0 = a1 
				a1 = z
				cost[t+2] = gz
				break

		if flag == 0:
			t = t-1
			break
		else:
			delta[t+2] = (1+sqrt(1+4*delta[t+1]**2))/2

		if abs( cost[t+2] - cost[t+1])/n <= 1e-2:
			break

	a = a1 
	cost = cost/n
	return a, cost


def repmat( data, r, c ):
	return kron( ones(( r, c)), data)


class KSH(object):

	def __init__(self, l, m, r, traindata, trainlabel, kernel):
		# np.random.seed(47)
		self.l = l
		self.m = m
		self.r = r
		self.kernel = kernel
		self.n = len(traindata)
		self.traindata = traindata
		self.traingnd = trainlabel

		print('Init the ksh...')
		# Proceed training sample set
		label_index = range(len(traindata))
		random.shuffle(label_index)
		self.label_index = array(label_index[:self.l], dtype=int32)

		sample_index = np.copy(self.label_index)
		random.shuffle(sample_index)
		self.sample = sample_index[:self.m]
		self.sampledata = self.traindata[self.sample]

		# self.traindata, self.traingnd = data_preprocessing(traindata)

		# number of anchors
		self.m = self.sample.shape[0]
		# number of labeled training samples
		self.trn = self.label_index.shape[0]

		# learn argument
		self.mvec, self.A = self.ksh(r)

		self.A = self.A[:,:r]

		del self.traindata
		del self.traingnd
		del self.label_index

	def kernel_vec(self, data):
		k_vec = self.kernel(data, self.sampledata)
		#for j in xrange(self.m):
		#	k_vec[j] = self.kernel.k(data, self.traindata[self.sample[j]])
		return k_vec - self.mvec

	def ksh(self, bit_num):
		# kernel computing
		# tn = self.testdata.shape[0]
		# KTrain = sqdist(self.traindata.T, anchor.T) #square distance between anchor and training datas
		# sigma  = mean(KTrain, axis=1).mean(axis=0)
		# KTrain = exp(-KTrain/(2*sigma)) #kernel matrix
		KTrain = self.kernel(self.traindata, self.sampledata)
		# KTrain = np.zeros((self.n, self.m))
		#for i in xrange(self.n):
		#	for j in xrange(self.m):
		#		KTrain[i,j] = self.kernel.k(self.traindata[i], self.traindata[self.sample[j]])
		mvec   = mean(KTrain, axis=0)
		KTrain = KTrain - repmat(mvec, self.traingnd.shape[0], 1) #Kernel matrix between anchor data and training datas, size=L*M

		#pairwise label matrix
		trngnd = mat(self.traingnd[self.label_index]).T
		temp   = repmat(trngnd, 1, self.trn) - repmat(trngnd.T, self.trn, 1)
		S0	 = -ones( (self.trn, self.trn), dtype=float32)
		tep	= (temp == 0).nonzero()
		S0[tep] = 1
		S = bit_num*S0


		# projection optimization
		KK = KTrain[self.label_index] #using part of data to learn parameters A
		RM = dot(KK.T,KK) 
		A1 = zeros( (self.m, bit_num), dtype=float32) #matrix A
		flag = zeros( bit_num )
		
		for rr in range(0,bit_num):
			print 'No: %d'%rr
			if rr > 0:
				S = S- dot(y, y.T) #initialize: R0=rS(bit_num*S0)

			#step 1: Spectral relaxation
			LM = dot( dot(KK.T, S), KK )
			(V, U) = eigh( LM, RM, eigvals_only=False)

			
			A1[:,rr] = U[:,(self.m-1)]
			tep = dot(dot( A1[:,rr].T, RM ), A1[:,rr])
			A1[:,rr] = sqrt(self.trn/tep)*A1[:,rr]

			#step 2: sigmoid smmothing
			get_vec, cost = OptProjectionFast( KK, S, A1[:,rr], 500)

			#A1[:rr]: a_k^0, get_vec: a_k^*
			y = dot(KK, A1[:,rr])
			y = (y> 0).choose( y, 1)
			y = (y<=0).choose( y,-1)
			y = y.reshape(y.shape[0], 1)

			y1 = dot(KK, get_vec)
			y1 = (y1>0).choose(y1,1)
			y1 = (y1<=0).choose(y1,-1)
			y1 = y1.reshape(y1.shape[0], 1)

			if dot(dot(y1.T, S), y1) > dot( dot(y.T,S), y):
				flag[rr] = 1
				A1[:,rr] = get_vec
				y = y1

		return mvec, A1

	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.sampledata)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.A)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)
