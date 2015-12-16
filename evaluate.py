import numpy as np
from data import hash_evaluation, multi_evaluation
import cPickle as cp
import time
from sdh import SDH
from dksh import DKSHv2
from ksh import KSH
from loader import Cifar10Loader, Cifar100Loader, NuswideLoader


def dump(filename, obj):
	with open(filename, 'wb') as f:
		cp.dump(obj, f)


def load(filename):
	with open(filename, 'rb') as f:
		return cp.load(f)


def eva_checkpoint(algo_name, nbit, li_results):
	dump('results_nus/{}_{}_step'.format(algo_name, nbit), li_results)
	res = multi_evaluation(li_results)
	dump('results_nus/{}_{}_total'.format(algo_name, nbit), res)
	if len(li_results) >= 1:
		print 'mean: mAP={}, pre2={}'.format(res['map_mean'], res['pre2_mean'])
	if len(li_results) >= 2:
		print 'std: mAP={}, pre2={}'.format(res['map_std'], res['pre2_std'])


def hash_factory(algo_name, nbits, nlabels, nanchors):
	if algo_name == 'SDH':
		return SDH(nbits, nanchors, nlabels, RBF)
	if algo_name == 'DKSH':
		return DKSHv2(nbits, nanchors, nlabels, RBF)
	if algo_name == 'KSH':
		return KSH(nbits, nanchors, nlabels, RBF)
	return None


def test(list_algo_name, list_bits, loader):
	seeds = [7, 17]#, 37, 47, 67, 97, 107, 127, 137, 157]
	for algo_name in list_algo_name:
		for nbit in list_bits:
			print '======execute {} at bit {}======'.format(algo_name, nbit)
			print '====total process round: {}====='.format(len(seeds))
			li_results = []
			for sd in seeds:
				print '\nround #{}...'.format(len(li_results)+1)

				traindata, trainlabel, basedata, baselabel, testdata, testlabel = loader.split(sd)

				alg = hash_factory(algo_name, nbit, 21, 1000)

				tic = time.clock()
				alg.train(traindata, trainlabel)
				toc = time.clock()
				print 'time:', toc-tic

				H_test = alg.queryhash(testdata)
				H_base = alg.queryhash(basedata)

				# make labels
				#gnd_truth = np.array([y == baselabel for y in testlabel]).astype(np.int8)
				gnd_truth = (np.dot(testlabel, baselabel.T) >= 1).astype(np.int8)

				print 'testing...'

				res = hash_evaluation(H_test, H_base, gnd_truth, 5000, topN=5000, trn_time=toc-tic)

				li_results.append(res)
				eva_checkpoint(algo_name, nbit, li_results)


def RBF(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.exp((2*np.dot(X,Y.T) - X2 - Y2)/1600)


if __name__ == "__main__":
	# init random seed

	# load data
	loader = NuswideLoader()

	# load algorithms
	list_algo_name = ['SDH', 'DKSH', 'KSH']
	list_nbits = [32]

	# test
	test(list_algo_name, list_nbits, loader)