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
	for i in xrange(81):
		trn_file = label_file[i*2]
		tst_file = label_file[i*2+1]


if __name__ == "__main__":
	#dat2npy('Low_Level_Features/BoW_Train_int.dat','bow_train.npy')
	#dat2npy('Low_Level_Features/BoW_Test_int.dat','bow_test.npy')
	dat2npy