import numpy as np
import matplotlib.pyplot as plt

sz = 100000

X = np.random.normal(size=(sz,64))
X /= np.linalg.norm(X, axis=1).reshape((sz,1))
X *= 8

Y = np.random.normal(size=(sz,64))
Y /= np.linalg.norm(X, axis=1).reshape((sz,1))
Y *= 8

for i in xrange(sz/2):
	d = float(i) / (sz/2) * 120
	r = np.sqrt((1024/(128-d))**2-64)
	Y[i] -= np.dot(Y[i], X[i])*X[i]
	Y[i] *= (r / np.linalg.norm(Y[i]))
	Y[i] += X[i]
	Y[i] *= (8.0 / np.linalg.norm(Y[i]))
	Y[sz-1-i] = -Y[i]

print 'make data success...'
dist = (np.sum(X*Y, axis=1)+64)/2
Hx = np.where(X>=0, 1, -1)
Hy = np.where(Y>=0, 1, -1)
ham = (np.sum(Hx*Hy, axis=1)+64)/2
print 'ploting...'
plt.plot(dist, ham, '.')
plt.show()