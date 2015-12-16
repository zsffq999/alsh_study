import numpy as np
import os


# save BoW
def dat2npy(src, dst):
	npybow = []
	with open(src, 'r') as f:
		for line in f:
			tmp = line.strip().split()
			npybow.append([int(a) for a in tmp])
	npybow = np.array(npybow, dtype=np.float32)
	np.save(dst, npybow)


def make_labels():
	label_file = sorted(os.listdir('TrainTestLabels'))
	train_labels = []
	test_labels = []
	for i in xrange(81):
		if i % 10 == 0:
			print i
		trn_file = label_file[i*2+1]
		tst_file = label_file[i*2]
		with open('TrainTestLabels/'+trn_file, 'r') as f:
			tmp = f.read().strip().split()
			train_labels.append([int(a) for a in tmp])
		with open('TrainTestLabels/'+tst_file, 'r') as f:
			tmp = f.read().strip().split()
			test_labels.append([int(a) for a in tmp])
	train_labels = np.array(train_labels, dtype=np.int32).T
	test_labels = np.array(test_labels, dtype=np.int32).T
	# find 21 most frequent labels
	num_labels = np.sum(train_labels, axis=0) + np.sum(test_labels, axis=0)
	idx = np.argsort(num_labels)[-21:]
	np.save('label_train.npy', train_labels[:,idx])
	np.save('label_test.npy', test_labels[:,idx])


if __name__ == "__main__":
	#dat2npy('Low_Level_Features/BoW_Train_int.dat','bow_train.npy')
	#dat2npy('Low_Level_Features/BoW_Test_int.dat','bow_test.npy')
	make_labels()