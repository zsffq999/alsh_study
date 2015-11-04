import numpy as np


def AsyMatFac(n, r, numlabel, mu):
	def objEvaluate(flag=1):
		if flag == 1:
			Tmp = np.dot(PH, QH.T)
			obj_1 = np.sum(Tmp*Tmp) / r / r
			obj_2 = n * np.sum((PH[:,l:]+QH[:,l:])**2)
			print "obj_1: {}, obj_2: {}, ave diff: {}, total: {}".format(obj_1, obj_2, obj_2/(4*n*n), obj_1+mu*obj_2)

	_mu = mu*r*r*n
	l = numlabel + 1
	# make label
	label = np.zeros(n, dtype=np.int32)
	for i in xrange(numlabel):
		label[int(float(n)/numlabel*i):int(float(n)/numlabel*(i+1))] = i
	# np.random.shuffle(label)
	print label

	# make similarity matrix
	# PH = [P, +H], QH = [Q, -H]
	PH = np.zeros((n,l+r), dtype=np.float32)
	PH[np.arange(n, dtype=np.int32), label] = 1
	QH = np.copy(PH)
	PH[:,:l-1] *= 2*r
	PH[:,l-1] = r
	QH[:,l-1] = -1

	# init: random generate each bit: each bit should balance.
	idx = np.arange(n, dtype=np.int32)
	for rr in xrange(r):
		np.random.shuffle(idx)
		h = np.ones(n)
		h[idx[:n/2]] = -1.0
		PH[:,l+rr] = h
		QH[:,l+rr] = -h

	objEvaluate()

	# do asymmetric loops
	hp = np.zeros(n)
	hq = np.zeros(n)
	for t in xrange(10):
		for rr in range(r):
			hp[:] = PH[:,l+rr]
			QH[:,l+rr] = PH[:,l+rr] = 0
			hq = np.dot(QH, np.dot(PH.T, hp)) + _mu*hp
			hq = np.where(hq>=0, -1, 1)
			PH[:,l+rr] = hp
			QH[:,l+rr] = hq

		for rr in range(r):
			hq[:] = QH[:,l+rr]
			QH[:,l+rr] = PH[:,l+rr] = 0
			hp = np.dot(PH, np.dot(QH.T, hq)) + _mu*hq
			hp = np.where(hp>=0, -1, 1)
			PH[:,l+rr] = hp
			QH[:,l+rr] = hq

		objEvaluate()

	Hp = PH[:,l:]
	Hq = -QH[:,l:]
	np.save('hp.npy', Hp)
	np.save('hq.npy', Hq)
	np.save('arglabel.npy', np.argsort(label).squeeze())


if __name__ == '__main__':
	AsyMatFac(5000, 32, 10, 0)