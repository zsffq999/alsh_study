import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix


class BQP(object):
	'''
	Definition interface of binary quadratic programming(BQP) problem.
	the problem is:
	max_{x={-1,1}^n} x.T*Q*x + q.T*x
	s.t. x \in S
	'''
	def __init__(self):
		pass

	def size(self):
		'''
		problem complexity
		'''
		return self.Y.shape[0]

	def neg_obj(self, x):
		'''
		neg objective given b
		'''
		raise NotImplementedError()

	def neg_grad(self, x):
		'''
		the gradient given b
		'''
		raise NotImplementedError()

	def max_eigen(self):
		'''
		the maximum eigenvalue of Q
		'''
		raise NotImplementedError()

	def col_sum(self, col_ind):
		'''
		compute the column sum of Q
		used in clustering method
		'''
		raise NotImplementedError()

	def diag(self):
		'''
		get the diagonal value of Q
		used in clustering method
		'''
		raise NotImplementedError()


class AMF_BQP(BQP):
	def __init__(self, Y, a, b, H, q=None):
		'''
		Q = a*Y*Y.T - b*1_{n*n} - H*H.T
		q = q
		'''
		super(AMF_BQP, self).__init__()
		self.Y = Y
		self.a = a
		self.b = b
		self.H = H
		if q == None:
			self.q = np.zeros(Y.shape[0])

	def neg_obj(self, x):
		return (-self.a)*np.sum(self.Y.T.dot(x)**2) - self.b*np.sum(x)**2 + np.sum(np.dot(x,self.H)**2) - np.dot(self.q, x)

	def neg_grad(self, x):
		return (-2*self.a)*self.Y.dot(self.Y.T.dot(x)) - (2*self.b)*np.sum(x) + 2*np.dot(self.H,np.dot(x,self.H)) - self.q

	def max_eigen(self):
		P = np.zeros((self.Y.shape[0], self.Y.shape[1]+1+self.H.shape[1]))
		P[:,:self.Y.shape[1]] = np.sqrt(np.abs(self.a))*self.Y[:,:self.Y.shape[1]].toarray()
		P[:,self.Y.shape[1]] = np.sqrt(np.abs(self.b))
		P[:,self.Y.shape[1]+1:] = self.H
		s, v, d = np.linalg.svd(P, full_matrices=False)
		d1 = np.copy(d)
		if self.a < 0:
			d1[:,:self.Y.shape[1]] = -d1[:,:self.Y.shape[1]]
		if self.b < 0:
			d1[:,self.Y.shape[1]] = -d1[:,self.Y.shape[1]]
		d1[:,self.Y.shape[1]+1:] = -d1[:,self.Y.shape[1]+1:]
		_v, _s = np.linalg.eig(np.dot(v.reshape((v.shape[0],1))*d, (v.reshape((v.shape[0],1))*d1).T))
		# s1, v1, d1 = np.linalg.svd(2*r*Y.dot(Y.T).toarray()-r-np.dot(H,H.T))
		ind = np.argmax(_v)
		# print np.min(_v)
		# print _v
		return np.dot(s, _s[:,ind])

	def col_sum(self, col_ind):
		n_tmp = len(col_ind)
		Y_tmp = self.Y[col_ind].sum(axis=0).A[0]
		return self.a*self.Y.dot(Y_tmp) + self.b * n_tmp

	def diag(self):
		return (self.a*self.Y.multiply(self.Y).sum(axis=1)).A[:,0] + self.b


def bqp_relax(bqp, init=None):
	'''
	binary quadratic programming with relaxation method
	:return: optimization value
	'''
	n = bqp.size()
	if None == init:
		init = np.where(np.random.rand(n)>0.5, 1, -1)
	bnds = [(-1,1) for i in xrange(n)]
	res = minimize(bqp.neg_obj, init, method='L-BFGS-B', jac=bqp.neg_grad, bounds=bnds, options={'disp': False, 'maxiter':500, 'maxfun':500})
	return np.where(res.x>0, 1, -1)


def bqp_spec(bqp, init=None):
	'''
	binary quadratic programming with spectual method
	:return:
	'''
	vs = bqp.max_eigen()
	# print bqp.neg_obj(vs)
	return np.where(vs>0, 1, -1)


def bqp_cluster(bqp, init=None):
	'''
	binary quadratic programming with clustering
	:return:
	'''
	n = bqp.size()
	maxiter = 10
	q = 0.5*bqp.q - bqp.col_sum(range(n))
	diag = bqp.diag()
	lamb = 0
	lamb_diag = lamb - diag

	# initialize cluster
	if None == init:
		cluster = np.argwhere(np.random.rand(n)>0.5)[:,0]
	else:
		cluster = np.argwhere(init==1)[:,0]

	# find final cluster
	for i in xrange(maxiter):
		dist_to_cluster = bqp.col_sum(cluster) + 0.5*q
		dist_to_cluster[cluster] += lamb_diag[cluster]
		# print i, dist_to_cluster
		med = np.median(dist_to_cluster)
		new_cluster = np.argwhere(dist_to_cluster>med)[:,0]
		if np.all(cluster == new_cluster):
			break
		cluster = new_cluster
	res = -np.ones(n)
	res[cluster] = 1
	return res


if __name__ == '__main__':
	n = 5000
	numlabel = 10
	r = 32
	trainlabel = np.random.randint(0,10, size=n)
	Y = csc_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,10), dtype=np.float32)
	H = np.where(np.random.rand(n,r)>0.5, 1, -1)

	def obj():
		Tmp = 2*Y.dot(Y.T).toarray()-1-np.dot(H,H.T)/float(r)
		return np.sum(np.sum(Tmp*Tmp, axis=1), axis=0)

	h = np.zeros(n)
	import time
	tic = time.clock()
	for t in xrange(3):
		for rr in xrange(r):
			h[:] = H[:,rr]
			H[:,rr] = 0
			bqp = AMF_BQP(Y, 2*r, -r, H)
			h1 = bqp_cluster(bqp, h)
			# print h1
			if bqp.neg_obj(h1) < bqp.neg_obj(h) or t == 0:
				H[:,rr] = h1
			else:
				H[:,rr] = h
			# print 'iter: {}, rr: {}, obj: {}'.format(t, rr, obj())
	toc = time.clock()
	print 'obj: {}; time: {}'.format(obj(), toc-tic)