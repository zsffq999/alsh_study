import numpy as np
import sys

sys.path.extend(['../'])

from data import hash_evaluation, multi_evaluation, batch_eva_ensem
import cPickle as cp
import time
from sdh import SDH
from dksh import DKSHv2
from ksh import KSH


def dump(filename, obj):
	with open(filename, 'wb') as f:
		cp.dump(obj, f)


def load(filename):
	with open(filename, 'rb') as f:
		return cp.load(f)


def eva_checkpoint(algo_name, nbit, li_results):
	dump('results_imagenet/{}_{}_step'.format(algo_name, nbit), li_results)
	res = multi_evaluation(li_results)
	dump('results_imagenet/{}_{}_total'.format(algo_name, nbit), res)
	if len(li_results) >= 1:
		print 'mean: mAP={}, pre2={}'.format(res['map_mean'], res['pre2_mean'])
	if len(li_results) >= 2:
		print 'std: mAP={}, pre2={}'.format(res['map_std'], res['pre2_std'])


def batch_checkpoint(algo_name, nbit, li_results):
	dump('results_imagenet/{}_{}_batch'.format(algo_name, nbit), li_results)
	res = batch_eva_ensem(li_results)
	print 'mean: mAP={}, pre2={}'.format(res['map'], res['pre2'])


def hash_factory(algo_name, nbits, nlabels, nanchors):
	if algo_name == 'SDH':
		return SDH(nbits, nanchors, nlabels, RBF)
	if algo_name == 'DKSH':
		return DKSHv2(nbits, nanchors, nlabels, RBF)
	if algo_name == 'KSH':
		return KSH(nbits, nanchors, nlabels, RBF)
	return None


def test(list_algo_name, list_bits):
	root = '../../ILSVRC2012_caffe/npy_data/'
	baselabel = np.load(root+'trainlabel.npy')
	testlabel = np.load(root+'testlabel.npy')
	testdata = np.load(root+'testdata_fc6_norm.npy')
	basicdata = np.load(root+'traindata_fc6_norm_1.npy')

	seeds = [7]#, 17, 37, 47, 67, 97, 107, 127, 137, 157]
	for algo_name in list_algo_name:
		for nbit in list_bits:
			print '======execute {} at bit {}======'.format(algo_name, nbit)
			print '====total process round: {}====='.format(len(seeds))
			li_results = []
			for sd in seeds:
				print '\nround #{}...'.format(len(li_results)+1)

				# load data
				np.random.seed(sd)
				idx = np.arange(100000, dtype=np.int32)
				np.random.shuffle(idx)

				traindata = basicdata[idx]
				trainlabel = baselabel[idx]

				alg = hash_factory(algo_name, nbit, 1000, 1000)

				tic = time.clock()
				alg.train(traindata, trainlabel)
				toc = time.clock()
				print 'time:', toc-tic

				print 'hash testing data...'
				H_test = alg.queryhash(testdata)
				print 'hash training data...'
				H_base = np.zeros((baselabel.shape[0], H_test.shape[1]), dtype=np.uint8)
				pt = 0
				for i in xrange(13):
					print 'hashing', i, '...'
					X = np.load('traindata_fc6_norm_{}.npy'.format(i+1))
					n = X.shape[0]
					H_base[pt:pt+n] = alg.basehash(X)
					pt += n

				print 'testing...'
				batch_results = []
				for i in xrange(50):
					# make labels
					print 'testing batch', i, '...'
					gnd_truth = np.array([y == baselabel for y in testlabel[i*1000:(i+1)*1000]]).astype(np.int8)
					batch_results.append(hash_evaluation(H_test, H_base, gnd_truth, 0, topN=5000, trn_time=toc-tic))
					eva_checkpoint(algo_name, nbit, batch_results)

				li_results.append(batch_eva_ensem(batch_results))
				eva_checkpoint(algo_name, nbit, li_results)


def RBF(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.exp(2*np.dot(X,Y.T) - X2 - Y2)


if __name__ == "__main__":
	# init random seed

	# load algorithms
	list_algo_name = ['SDH', 'DKSH', 'KSH']
	list_nbits = [32]

	# test
	test(list_algo_name, list_nbits)