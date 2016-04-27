from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

class CNN_hash(object):
	def __init__(self, solver_proto, device=0):
		self.solver_proto = solver_proto
		self.device = device
		self.model_dir = ""
		self.out_blob = ""

	def train(self, traindata, H):
		caffe.set_device(self.device)
		caffe.set_mode_gpu()
		self.solver = None
		self.solver = caffe.SGDSolver(self.solver_proto)
		if self.model_dir != "":
			self.solver.net.copy_from(self.model_dir)
			self.solver.test_net[0].share_with(self.solver)
		self.solver.net.set_input_arrays(traindata, H)
		self.solver.solve()

	def predict(self, traindata):
		self.solver.net.blobs[self.out_blob]