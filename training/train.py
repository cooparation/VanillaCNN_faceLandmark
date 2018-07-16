import sys
sys.path.append('../caffe/python')
import caffe
sys.path.append('./common')
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('training/net2/vanilla_adam_solver.prototxt')
solver.solve()
