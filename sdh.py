import numpy as np
from data import hash_value, hash_evaluation
from ksh import RBF
import time


class SDH(object):
	def __init__(self, r, m, numlabel, kernel):
		self.r = r
		self.m = m
		self.numlabel = numlabel
		self.kernel = kernel

		self.lamb = 1
		self.mu = 1e-5

		self.mvec = None
		self.B = None
		self.P = None

	def train(self, traindata, trainlabel):
		# for debugging
		def evaObjective(flag):
			if flag == 1:
				obj_G = np.linalg.norm(Y-np.dot(B, W)) + self.lamb*np.linalg.norm(W)
				obj_F = np.linalg.norm(B-np.dot(KK,P))
				print 'Obj Value: obj_G={}, obj_F={}, total={}'.format(obj_G, obj_F, obj_G+self.mu*obj_F)

		n = len(traindata)
		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]
		trainlabel = trainlabel[indexes]
		self.trainlabel = trainlabel

		# determine anchors
		anchoridx = np.copy(indexes)
		np.random.shuffle(anchoridx)
		anchoridx = anchoridx[:self.m]
		self.anchors = traindata[anchoridx]

		# kernel matrix and mean
		KK = self.kernel(traindata, self.anchors)

		print KK
		print np.mean(KK)
		print np.median(KK)
		print np.max(KK)
		np.save('tmp.npy', KK)

		self.mvec = np.mean(KK, axis=0).reshape((1, self.m))
		KK = KK - self.mvec

		B = np.random.rand(n, self.r)
		B = np.where(B>0.5, 1, -1).astype(np.float32)

		Y = np.zeros((n,self.numlabel), dtype=np.float32)
		if len(trainlabel.shape) >= 2:
			assert trainlabel.shape[1] == self.numlabel
			Y[:,:self.numlabel] = trainlabel
		else:
			Y[np.arange(n, dtype=np.int32), trainlabel] = 1

		for tt in range(10):
			print 'iter:', tt

			# G-step: compute W
			W = np.dot(np.linalg.inv(np.dot(B.T, B)+self.lamb*np.eye(self.r)), np.dot(B.T, Y))

			# F-step: compute P
			P_L = np.dot(np.linalg.inv(np.dot(KK.T, KK)), KK.T)
			P = np.dot(P_L, B)

			if tt > 0:
				if np.linalg.norm(P0-P) < 1e-5*np.linalg.norm(P0):
					break
			P0 = np.copy(P)

			evaObjective(1)

			# B-step: compute B by DCC
			Q = np.dot(Y, W.T) + self.mu * np.dot(KK, P)
			B_W = np.zeros((n, self.numlabel))
			B = np.zeros((n, self.r), dtype=np.float32)
			Z = np.copy(B)
			for t in range(10):
				Z[:,:] = B
				for rr in range(self.r):
					B_W -= np.dot(B[:,rr:rr+1], W[rr:rr+1,:])
					z = Q[:,rr] - np.dot(B_W, W[rr,:])
					z = np.where(z>=0, 1, -1)
					B[:,rr] = z
					B_W += np.dot(B[:,rr:rr+1], W[rr:rr+1,:])
				if np.linalg.norm(B-Z) < 1e-5*np.linalg.norm(B):
					break

			evaObjective(1)

			if np.linalg.norm(B-np.dot(KK,P)) < 1e-5*np.linalg.norm(B):
				break

		# Finally, do F-step one step more: compute P
		P_L = np.dot(np.linalg.inv(np.dot(KK.T, KK)), KK.T)
		P = np.dot(P_L, B)

		self.B = B
		self.P = P

	def queryhash(self, qdata):
		Kdata = self.kernel(qdata, self.anchors)
		Kdata -= self.mvec
		Y = np.dot(Kdata, self.P)
		Y = np.where(Y>=0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		H = np.where(self.B>=0, 1, 0)
		return hash_value(H)


def test():
	#np.random.seed(47)
	X = np.load('data/cifar_gist.npy')
	Y = np.load('data/cifar_label.npy')

	traindata = X[:59000]
	trainlabel = Y[:59000]
	basedata = X[:59000]
	baselabel = Y[:59000]
	testdata = X[59000:]
	testlabel = Y[59000:]


	# train model
	sdh = SDH(32, 300, 10, RBF)
	tic = time.clock()
	sdh.train(traindata, trainlabel)
	toc = time.clock()
	print 'time:', toc-tic

	H_test = sdh.queryhash(testdata)
	H_base = sdh.queryhash(basedata)

	idx = np.argsort(sdh.trainlabel).squeeze()
	bb = sdh.B[idx[0:59000:6000]]
	classham = (32-np.dot(bb,bb.T))/2
	print np.sum(classham)
	print classham

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']

if __name__ == "__main__":
	test()