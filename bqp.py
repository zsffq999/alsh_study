import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix, kron, vstack


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
		Q = a*Y*Y.T + b*1_{n*n} - H*H.T
		q = q
		'''
		super(AMF_BQP, self).__init__()
		self.Y = Y
		self.a = a
		self.b = b
		self.H = H
		if q is None:
			self.q = np.zeros(Y.shape[0])
		else:
			self.q = q

	def neg_obj(self, x):
		return (-self.a) * np.sum(self.Y.T.dot(x) ** 2) - self.b * np.sum(x) ** 2 + np.sum(
			np.dot(x, self.H) ** 2) - np.dot(self.q, x)

	def neg_grad(self, x):
		return (-2 * self.a) * self.Y.dot(self.Y.T.dot(x)) - (2 * self.b) * np.sum(x) + 2 * np.dot(self.H, np.dot(x,self.H)) - self.q

	def max_eigen(self):
		P = np.zeros((self.Y.shape[0], self.Y.shape[1] + 1 + self.H.shape[1]))
		P[:, :self.Y.shape[1]] = np.sqrt(np.abs(self.a)) * self.Y[:, :self.Y.shape[1]].toarray()
		P[:, self.Y.shape[1]] = np.sqrt(np.abs(self.b))
		P[:, self.Y.shape[1] + 1:] = self.H
		s, v, d = np.linalg.svd(P, full_matrices=False)
		d1 = np.copy(d)
		if self.a < 0:
			d1[:, :self.Y.shape[1]] = -d1[:, :self.Y.shape[1]]
		if self.b < 0:
			d1[:, self.Y.shape[1]] = -d1[:, self.Y.shape[1]]
		d1[:, self.Y.shape[1] + 1:] = -d1[:, self.Y.shape[1] + 1:]
		_v, _s = np.linalg.eig(np.dot(v.reshape((v.shape[0], 1)) * d, (v.reshape((v.shape[0], 1)) * d1).T))
		# s1, v1, d1 = np.linalg.svd(2*r*Y.dot(Y.T).toarray()-r-np.dot(H,H.T))
		ind = np.argmax(_v)
		# print np.min(_v)
		# print _v
		return np.dot(s, _s[:, ind])

	def col_sum(self, col_ind):
		n_tmp = len(col_ind)
		Y_tmp = self.Y[col_ind].sum(axis=0).A[0]
		return self.a * self.Y.dot(Y_tmp) + self.b * n_tmp - np.dot(self.H, np.sum(self.H[col_ind], axis=0))

	def diag(self):
		return (self.a * self.Y.multiply(self.Y).sum(axis=1)).A[:, 0] + self.b


class AMF_deg3_BQP(BQP):
	def __init__(self, Y, a, b, c, d, H, q=None):
		'''
		Q_{i,j} = a*(y_i*y_j)^3+b*(y_i*y_j)^2+c*(y_i+y_j) + d - h_i*h_j
		q = q
		'''
		super(AMF_deg3_BQP, self).__init__()
		n, l =  Y.shape
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.Y1 = Y
		self.H = H # mostly updated
		if q is None:
			self.q = np.zeros(n)
		else:
			self.q = q
		# consider 2nd and 3nd power of Y
		Y2_tmp = []
		Y3_tmp = []
		for i in xrange(n):
			y = Y.getrow(i)
			Y2_tmp.append(kron(y, y))
			Y3_tmp.append(kron(kron(y, y), y))
		self.Y2 = vstack(Y2_tmp).tocsr()
		self.Y3 = vstack(Y3_tmp).tocsr()

	def size(self):
		return self.Y1.shape[0]

	def neg_obj(self, x):
		return (-self.a) * np.sum(self.Y3.T.dot(x) ** 2) -self.b * np.sum(self.Y2.T.dot(x) ** 2) - self.c * np.sum(self.Y1.T.dot(x) ** 2)\
			- self.d * np.sum(x) ** 2 + np.sum(np.dot(x, self.H) ** 2) - np.dot(self.q, x)

	def neg_grad(self, x):
		return (-2 * self.a) * self.Y3.dot(self.Y3.T.dot(x)) - (2 * self.b) * self.Y2.dot(self.Y2.T.dot(x)) - \
			(2 * self.c) * self.Y1.dot(self.Y1.T.dot(x)) - (2 * self.d) * np.sum(x) + 2 * np.dot(self.H, np.dot(x,self.H)) - self.q

	def col_sum(self, col_ind):
		n_tmp = len(col_ind)
		Y1_tmp = self.Y1[col_ind].sum(axis=0).A[0]
		Y2_tmp = self.Y2[col_ind].sum(axis=0).A[0]
		Y3_tmp = self.Y3[col_ind].sum(axis=0).A[0]
		return self.a * self.Y3.dot(Y3_tmp) + self.b * self.Y2.dot(Y2_tmp) + self.c * self.Y1.dot(Y1_tmp)\
			 + self.d * n_tmp - np.dot(self.H, np.sum(self.H[col_ind], axis=0))

	def diag(self):
		return (self.a * self.Y3.multiply(self.Y3).sum(axis=1) + self.b * self.Y2.multiply(self.Y2).sum(axis=1)\
				+ self.c * self.Y1.multiply(self.Y1).sum(axis=1)).A[:, 0] + self.d


def bqp_relax(bqp, init=None):
	'''
	binary quadratic programming with relaxation method
	:return: optimization value
	'''
	n = bqp.size()
	if init is None:
		init = np.where(np.random.rand(n) > 0.5, 1, -1)
	bnds = [(-1, 1) for i in xrange(n)]
	res = minimize(bqp.neg_obj, init, method='L-BFGS-B', jac=bqp.neg_grad, bounds=bnds,
				options={'disp': False, 'maxiter': 500, 'maxfun': 500})
	return np.where(res.x > 0, 1, -1)


def bqp_spec(bqp, init=None):
	'''
	binary quadratic programming with spectual method
	:return:
	'''
	vs = bqp.max_eigen()
	# print bqp.neg_obj(vs)
	return np.where(vs > 0, 1, -1)


def bqp_cluster(bqp, init):
	'''
	binary quadratic programming with clustering
	:return:
	'''
	n = bqp.size()
	maxiter = 10
	q = 0.5 * bqp.q - bqp.col_sum(range(n))
	diag = bqp.diag()
	lamb = 320
	lamb_diag = lamb - diag

	# initialize cluster
	res = init
	cluster = np.argwhere(res >= 0)[:, 0]

	# find final cluster
	for i in xrange(maxiter):
		dist_to_cluster = bqp.col_sum(cluster) + 0.5 * q
		dist_to_cluster[cluster] += lamb_diag[cluster]
		# print i, dist_to_cluster
		med = np.median(dist_to_cluster)
		new_res = np.where(dist_to_cluster > med, 1, -1)
		# print np.sum(res==1), np.sum(new_res==1),np.sum(res != new_res)
		if np.sum(res != new_res) < 3:
			break
		cluster = np.argwhere(new_res >= 0)[:, 0]
		res = new_res
	res = new_res
	return res


def bqp_cluster_opt(bqp, init=None):
	'''
	bqp with clustering, k-means initialization
	:param bqp:
	:param init:
	:return:
	'''
	n = bqp.size()
	maxiter = 10
	# q = 0.5*bqp.q - bqp.col_sum(range(n))
	q = 0.5 * bqp.q
	diag = bqp.diag()
	lamb = 320
	lamb_diag = lamb - diag

	# initialize cluster
	res = init
	cluster_1 = np.argwhere(res >= 0)[:, 0]
	cluster_2 = np.argwhere(res < 0)[:, 0]

	# init cluster with K-means
	q_cluster_no = 1
	for i in xrange(maxiter):
		dist_to_cluster_1 = bqp.col_sum(cluster_1)
		dist_to_cluster_1[cluster_1] += lamb_diag[cluster_1]
		dist_to_cluster_2 = bqp.col_sum(cluster_2)
		dist_to_cluster_2[cluster_2] += lamb_diag[cluster_2]
		if q_cluster_no == 1:
			dist_to_cluster_1 += q
			dist_to_cluster_1 /= len(cluster_1) + 1
			dist_to_cluster_2 /= len(cluster_2)
			q_dist_1 = (np.sum(q[cluster_1])) / (len(cluster_1) + 1)
			q_dist_2 = np.mean(q[cluster_1])
		else:
			dist_to_cluster_2 += q
			dist_to_cluster_1 /= len(cluster_1)
			dist_to_cluster_2 /= len(cluster_2) + 1
			q_dist_2 = (np.sum(q[cluster_2])) / (len(cluster_2) + 1)
			q_dist_1 = np.mean(q[cluster_1])
		new_res = np.where(dist_to_cluster_1 >= dist_to_cluster_2, 1, -1)
		new_q_cluster_no = 1 if q_dist_1 >= q_dist_2 else 2
		if np.sum(res != new_res) + (new_q_cluster_no != q_cluster_no) <= 4:
			break
		res = new_res
		cluster_1 = np.argwhere(res >= 0)[:, 0]
		cluster_2 = np.argwhere(res < 0)[:, 0]
		q_cluster_no = new_q_cluster_no

	if np.sum(dist_to_cluster_1) + q_dist_1 < np.sum(dist_to_cluster_2) + q_dist_2:
		res = -res

	# find final cluster
	q -= 0.25 * bqp.col_sum(range(n))
	cluster = np.argwhere(res >= 0)[:, 0]

	for i in xrange(maxiter):
		dist_to_cluster = bqp.col_sum(cluster) + q
		dist_to_cluster[cluster] += lamb_diag[cluster]
		# print i, dist_to_cluster
		med = np.median(dist_to_cluster)
		new_res = np.where(dist_to_cluster > med, 1, -1)
		# print np.sum(res==1), np.sum(new_res==1),np.sum(res != new_res)
		if np.sum(res != new_res) < 3:
			break
		cluster = np.argwhere(new_res >= 0)[:, 0]
		res = new_res
	res = new_res
	return res


def bqp_combine(bqp, init):
	h_tmp = bqp_cluster(bqp, init)
	h = bqp_relax(bqp, h_tmp)
	return h


def test():
	tot_time = []
	tot_loss = []
	for ttt in xrange(20):
		n = 5000
		numlabel = 10
		r = 32
		# generate multi labels
		n_ind = []
		l_ind = []
		labels = np.arange(numlabel, dtype=np.int32)
		for i in xrange(n):
			num = np.random.poisson() + 1
			if num > numlabel - 2:
				num = numlabel - 2
			# num = 1
			n_ind += [i] * num
			np.random.shuffle(labels)
			l_ind += labels[:num].tolist()

		Y = csc_matrix((np.ones(len(n_ind)), [n_ind, l_ind]), shape=(n, 10), dtype=np.float32)
		H = np.where(np.random.rand(n, r) > 0.5, 1, -1)

		def obj():
			Tmp = np.where(Y.dot(Y.T).toarray() >= 1, 1, -1) - np.dot(H, H.T) / float(r)
			return np.sum(np.sum(Tmp * Tmp, axis=1), axis=0)

		h = np.zeros(n)
		import time
		tic = time.clock()
		bqp = AMF_deg3_BQP(Y, 1.0/3*r, -2*r, 11.0/3*r, -r, H)
		for t in xrange(3):
			for rr in xrange(r):
				h[:] = H[:, rr]
				H[:, rr] = 0
				bqp.H = H
				# h1 = bqp_cluster(bqp, h)
				h1 = bqp_relax(bqp, h)
				# print h1
				if bqp.neg_obj(h1) < bqp.neg_obj(h) or t == 0:
					H[:, rr] = h1
				else:
					H[:, rr] = h
				# print 'iter: {}, rr: {}, obj: {}'.format(t, rr, obj())

		toc = time.clock()
		loss = obj()
		tot_time.append(toc - tic)
		tot_loss.append(loss)
		print 'round', ttt, toc - tic, loss
	print np.mean(tot_loss), np.std(tot_loss)
	print np.mean(tot_time), np.std(tot_time)


if __name__ == '__main__':
	test()
	'''
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
			h1 = bqp_cluster_opt(bqp, h)
			# print h1
			if bqp.neg_obj(h1) < bqp.neg_obj(h) or t == 0:
				H[:,rr] = h1
			else:
				H[:,rr] = h
			print 'iter: {}, rr: {}, obj: {}'.format(t, rr, obj())
	toc = time.clock()
	print 'obj: {}; time: {}'.format(obj(), toc-tic)
	'''
