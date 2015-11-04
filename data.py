import numpy as np
import time
import matplotlib.pyplot as plt


bit_num_in_uint8 = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, \
	4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, \
	2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, \
	4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3,\
	3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, \
	4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, \
	6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])


def hash_value(bitarray, unit=1):
	# unit: number of bit containing a hash value. default 1
	if unit == 1:
		return np.packbits(bitarray, axis=-1)
	else:
		bitarray = np.where(bitarray<=127, bitarray, 127)
		bitarray = np.where(bitarray>=-128, bitarray, -128)
		return bitarray.astype(np.int8)


def hamming_dist_mat(Y, X, unit=1):
	# unit: number of bit containing a hash value. default 1
	if unit == 1:
		return np.sum(bit_num_in_uint8[[np.bitwise_xor(y, X) for y in Y]], axis=-1)
	else:
		return np.array([np.sum(y != X, axis=-1) for y in Y])


def hash_evaluation(Y, X, GndTruth, topMAP=5000, topN=10000, unit=1):
	results = {}
	total = np.sum(GndTruth, dtype=np.int64)
	results['total'] = total
	results['leny'] = len(Y)

	# searching and computing searching time...
	tic = time.clock()
	# maybe we have to sort with batch X, but further on
	h_mat = hamming_dist_mat(Y, X, unit)
	ham_ranks = np.argsort(h_mat, axis=1)
	toc = time.clock()
	results['time'] = toc - tic

	# recall at top N
	rank_rec = np.sum([np.cumsum(GndTruth[i, ham_ranks[i,:topN]]) for i in xrange(len(Y))], axis=0)
	results['rec'] = rank_rec/float(total)

	# precision-recall
	pr = np.array([rank_rec/float(total), rank_rec/(np.arange(topN)+1.0)/len(Y)])
	results['pr'] = pr

	# mAP
	mAP = 0
	for i in xrange(len(Y)):
		rights = np.argwhere(GndTruth[i, ham_ranks[i]]==1)[:,0] + 1
		if i % 100 == 0:
			print rights[:10]
		# print rights
		rightcnt = np.searchsorted(rights, topMAP, 'r')
		if rightcnt == 0:
			continue
		mAP += np.mean((np.arange(rightcnt, dtype=np.float32)+1.0) / rights[:rightcnt])
	results['map'] = mAP / len(Y)

	return results


def batch_eva_ensem(li_results):
	results = {}

	results['time'] = sum([r['time'] for r in li_results])
	results['leny'] = sum(r['leny'] for y in li_results)

	totals = np.array([r['total'] for r in li_results])
	results['total'] = np.sum(totals, dtype=np.int64)

	results['map'] = np.sum(np.array([r['map']*r['total'] for r in li_results])) / results['total']

	rank_rec = np.sum(np.array([r['rec']*r['total']] for r in li_results), axis=0)
	results['rec'] = rank_rec / results['total']
	results['pr'] = rank_rec / (np.arange(len(rank_rec))+1.0) / results['leny']

	return results


def l2_dist_mat(X, Y):
	lenX = X.shape[0]
	lenY = Y.shape[0]
	X2 = np.dot(np.sum(X * X, axis=1).reshape((lenX, 1)), np.ones((1, lenY), dtype=np.float32))
	Y2 = np.dot(np.ones((lenX, 1), dtype=np.float32), np.sum(Y * Y, axis=1).reshape((1, lenY)))
	return np.sqrt(-2 * np.dot(X, Y.T) + X2 + Y2)


def l2_gnd_truth(Y, X, topk):
	distmat = l2_dist_mat(Y, X)
	ori_ranks = np.argsort(distmat, axis=1)[:,:topk]
	labelmat = np.zeros((len(Y), len(X)), dtype=np.int8)
	for i in xrange(len(Y)):
		labelmat[i, ori_ranks[i]] = 1
	return labelmat