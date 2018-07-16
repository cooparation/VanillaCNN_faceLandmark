import sys
sys.path.append('../caffe/python')
import caffe
sys.path.append('./train')
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('train/adam_solver_5p.prototxt')
solver.solve()
