import numpy as np

# preprossing of imagenet
# first of all, compute feature mean
mean = np.zeros((13,4096))
num = np.zeros(13)
for i in xrange(13):
	print 'computing', i, '...'
	data = np.load('traindata_fc6_{}.npy'.format(i+1))
	mean[i] = np.mean(data, axis=0)
	num[i] = data.shape[0]
num /= np.sum(num)
res_mean = np.dot(mean.T, num).reshape((1, 4096))
print res_mean

for i in xrange(13):
	print 'computing', i, '...'
	X = np.load('traindata_fc6_{}.npy'.format(i+1))
	X -= res_mean
	X /= np.linalg.norm(X, axis=1).reshape((len(X),1))
	np.save('traindata_fc6_norm_{}.npy'.format(i+1), X)

Y = np.load('testdata_fc6.npy')
Y -= res_mean
Y /= np.linalg.norm(Y, axis=1).reshape((len(Y),1))
np.save('testdata_fc6_norm.npy', Y)