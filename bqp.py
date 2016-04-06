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
		d1 = np.zeros(d.shape)
		if self.a < 0:
			d1[:,:self.Y.shape[1]] = -d[:,:self.Y.shape[1]]
		if self.b < 0:
			d1[:,self.Y.shape[1]] = -d[:,self.Y.shape[1]]
		d1[:,self.Y.shape[1]+1:] = -d[:,self.Y.shape[1]+1:]
		_s, _v, _ = np.linalg.svd(np.dot(v.reshape((v.shape[0],1))*d, (v.reshape((v.shape[0],1))*d1).T))
		print _v
		s1, v1, d1 = np.linalg.svd(2*r*Y.dot(Y.T).toarray()-r-np.dot(H,H.T))
		print v1[:40]
		return np.dot(s, _s[:,0])

	def col_sum(self, col_ind):
		return np.dot(self.P, np.sum(self.Q[col_ind], axis=0))

	def diag(self):
		return self.a*np.sum(self.Y*self.Y, axis=1) + self.b


def bqp_relax(bqp, init=None):
	'''
	binary quadratic programming with relaxation method
	:return: optimization value
	'''
	n = bqp.size()
	if init == None:
		init = np.where(np.random.rand(n)>0.5, 1, -1)
	bnds = [(-1,1) for i in xrange(n)]
	res = minimize(bqp.neg_obj, init, method='L-BFGS-B', jac=bqp.neg_grad, bounds=bnds, options={'disp': False, 'maxiter':500, 'maxfun':500})
	return np.where(res.x>0, 1, -1)


def bqp_spec(bqp, init=None):
	'''
	binary quadratic programming with spectual method
	:return:
	'''
	return np.where(bqp.max_eigen()>0, 1, -1)


def bqp_cluster(bqp, init=None):
	'''
	binary quadratic programming with clustering
	:return:
	'''
	n = bqp.size()
	maxiter = 10
	q = 0.5*bqp.q - bqp.col_sum(slice(n))
	diag = bqp.diag()
	lamb = n
	lamb_diag = lamb - diag

	# initialize cluster
	if init == None:
		cluster = np.argwhere(np.random.rand(n)>0.5)[:,0]
	else:
		cluster = np.argwhere(init==1)[:,0]

	# find final cluster
	for i in xrange(maxiter):
		dist_to_cluster = bqp.col_sum(cluster) + 0.5*q
		dist_to_cluster[cluster] += lamb_diag[cluster]
		med = np.median(dist_to_cluster)
		new_cluster = np.argwhere(dist_to_cluster>med)[:,0]
		if cluster == new_cluster:
			break
		cluster = new_cluster
	res = -np.ones(n)
	res[cluster] = 1
	return res


if __name__ == '__main__':
	n = 500
	numlabel = 10
	r = 32
	trainlabel = np.random.randint(0,10, size=n)
	Y = csc_matrix((np.ones(n),[np.arange(n, dtype=np.int32), trainlabel]), shape=(n,10), dtype=np.float32)
	H = np.where(np.random.rand(n,r)>0.5,1,-1)

	def obj():
		Tmp = 2*Y.dot(Y.T).toarray()-1-np.dot(H,H.T)/float(r)
		return np.sum(np.sum(Tmp*Tmp, axis=1), axis=0)


	h = np.zeros(n)
	for t in xrange(5):
		for rr in xrange(r):
			h[:] = H[:,rr]
			H[:,rr] = 0
			bqp = AMF_BQP(Y, 2*r, -r, H)
			h1 = bqp_spec(bqp, h)
			print h1
			if bqp.neg_obj(h1) < bqp.neg_obj(h):
				H[:,rr] = h1
			else:
				H[:,rr] = h
			print 'Iter {}, step {}: obj={}'.format(t, rr, obj())