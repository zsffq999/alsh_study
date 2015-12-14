import numpy as np
from data import hash_value, hash_evaluation
from scipy.optimize import minimize
from sklearn.svm import SVC
from ksh import RBF


class SHLE(object):
	'''
	Supervised hashing with label embedding
	'''
	def __init__(self, r, numlabel, ClasfierMkr):
		self.r = r
		self.numlabel = numlabel
		self.classifier = []
		self.mu = 0.1
		for i in xrange(self.r):
			self.classifier.append(ClasfierMkr())

	def train(self, traindata, trainlabel):
		def objEvaluate(flag=1):
			if flag == 1:
				print '----------evaluation----------'
				# precision
				print np.mean((P+1)/2, axis=1)
				# matrix loss
				print np.linalg.norm(R)**2 / self.r / self.r
				# hash matrix
				print (self.r-np.dot(H,H.T)) / 2

		n = len(traindata)
		mu = self.mu * self.numlabel

		# shuffle data
		indexes = np.arange(n, dtype=np.int32)
		np.random.shuffle(indexes)
		traindata = traindata[indexes]
		trainlabel = trainlabel[indexes]

		# split into training set and validation set
		_tdata = traindata[:n*4/5]
		_tlabel = trainlabel[:n*4/5]
		_vdata = traindata[n*4/5:]
		_vlabel = trainlabel[n*4/5:]

		n_count = np.bincount(_vlabel)
		print n_count

		# random init labeled hash matrix
		# H = np.ones((self.numlabel, self.r))
		H = np.random.rand(self.numlabel, self.r)
		H = np.where(H>0.5, 1, -1)

		# optimize relaxed function using coordinate descent
		R = np.dot(H, H.T) - self.r*(2*np.eye(self.numlabel) - np.ones((self.numlabel, self.numlabel)))
		fun = lambda x: np.dot(np.dot(x,R),x)
		gra = lambda x: 2*np.dot(R, x)
		bnds = [(-1,1) for i in xrange(self.numlabel)]
		for rr in xrange(self.r):
			R -= np.dot(H[:,rr:rr+1], H[:,rr:rr+1].T)
			res = minimize(fun, H[:,rr], method='L-BFGS-B', jac=gra, bounds=bnds, options={'disp': False})
			H[:,rr] = np.where(res.x>0, 1, -1)
			R += np.dot(H[:,rr:rr+1], H[:,rr:rr+1].T)

		# do 5 iterations
		for t in xrange(1):
			print 'iter', t, '...'
			# train hash function using classification algorithms
			P = np.zeros((self.numlabel, self.r))
			for rr in xrange(self.r):
				# train hash function
				_tmptrnlabel = H[_tlabel, rr].astype(np.int32)
				self.classifier[rr].fit(_tdata, _tmptrnlabel)

				# test accuracy of hash function
				_tmptstlabel = np.where(self.classifier[rr].predict(_vdata)>=0, 1, -1)
				_truetstlabel = H[_vlabel, rr].astype(np.int32)
				_res = (_tmptstlabel == _truetstlabel).astype(np.int32)
				P[:,rr] = np.bincount(_vlabel, _res) / n_count

			# adjust hash function using precision matrix P
			P *= 2
			P -= 1
			objEvaluate()

			for rr in xrange(self.r):
				p = P[:,rr]
				R -= np.dot(H[:,rr:rr+1], H[:,rr:rr+1].T)
				fun2 = lambda x: np.dot(np.dot(x,R), x) - mu * np.dot(p, x)
				gra2 = lambda x: 2*np.dot(R,x) - mu * p
				res = minimize(fun2, H[:,rr], method='L-BFGS-B', jac=gra2, bounds=bnds, options={'disp': False})
				H[:,rr] = np.where(res.x>0, 1, -1)
				R += np.dot(H[:,rr:rr+1], H[:,rr:rr+1].T)

		# do final iteration using whole training data
		for rr in xrange(self.r):
			# train hash function
			_tmptrnlabel = H[trainlabel, rr].astype(np.int32)
			self.classifier[rr].fit(traindata, _tmptrnlabel)

		objEvaluate()

	def queryhash(self, qdata):
		Y = np.zeros((len(qdata), self.r), dtype=np.int8)
		for rr in xrange(self.r):
			print rr
			res = self.classifier[rr].predict(qdata)
			Y[:,rr] = np.where(res>0, 1, 0)
		return hash_value(Y)

	def basehash(self, data):
		return self.queryhash(data)


class RBFClassifier():
	def __init__(self, anchors):
		self.anchors = anchors
		self.m = len(anchors)
		self.len = 0

	def fit(self, data, label):
		KK = RBF(data, self.anchors)
		self.mvec = np.mean(KK, axis=0).reshape((1, self.m))
		KK -= self.mvec
		RM = np.dot(KK.T, KK) + np.eye(self.m)
		invRM = np.linalg.inv(RM)
		self.w = np.dot(invRM, np.dot(KK.T, label))

	def predict(self, data):
		if self.len != len(data):
			self.len = len(data)
			self.Kdata = RBF(data, self.anchors) - self.mvec
		Y = np.dot(self.Kdata, self.w)
		return np.where(Y>=0, 1, -1)

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
	alg = SHLE(32, 10, lambda : RBFClassifier(X[:300]))
	alg.train(traindata, trainlabel)

	H_test = alg.queryhash(testdata)
	H_base = alg.queryhash(basedata)

	'''
	idx = np.argsort(alg.trainlabel).squeeze()
	bb = alg.H[idx[0:5000:530]]
	classham = (32-np.dot(bb,bb.T))/2
	print np.sum(classham)
	print classham
	'''

	# make labels
	gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)

	print 'testing...'

	res = hash_evaluation(H_test, H_base, gnd_truth, 59000)
	print 'MAP:', res['map']

if __name__ == "__main__":
	test()