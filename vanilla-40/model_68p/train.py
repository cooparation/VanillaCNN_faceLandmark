import sys
sys.path.append('../caffe/python')
import caffe
sys.path.append('./common')
caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('training/Model_68Point/vanilla_adam_solver.prototxt')
solver.solve()
