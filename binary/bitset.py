import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_size_t

array_2d_uint8 = npcd.ndpointer(dtype=np.uint8, ndim=2, flags='CONTIGUOUS')
array_2d_uint32 = npcd.ndpointer(dtype=np.uint32, ndim=2, flags='CONTIGUOUS')

libcd = npct.load_library('xxx', '.')

libcd.hamming_rank.restype = None
libcd.hamming_rank.argtypes = [c_size_t, array_2d_uint8, c_size_t, array_2d_uint8, c_size_t, array_2d_uint32]

def hamming_rank(Y, X):
	assert Y.shape[1] == X.shape[1]
	res = np.zeros((len(Y), len(X)), dtype=np.int32)
	libcd.hamming_rank(8*Y.shape[1], Y, len(Y), X, len(X), res)
	return res