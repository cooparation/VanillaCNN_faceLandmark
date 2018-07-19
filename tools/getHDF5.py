import sys
sys.path.append("../caffe/python")
import h5py, os
import caffe
import numpy as np
 
SIZE = 60 # fixed size to all images
NUM_POINTS=10

#with open( './data/train.txt', 'r' ) as T :
with open( './data/test.txt', 'r' ) as T :
     lines = T.readlines()
     data_ = np.zeros( (len(lines), 3, SIZE, SIZE), dtype='f4' )
     label_ = np.zeros( (len(lines), NUM_POINTS), dtype='f4' )

for i,l in enumerate(lines):
     sp = l.strip().split()
     img = caffe.io.load_image(sp[0] )
     height,width =  img.shape[0],img.shape[1]
     print height, width
 
     img = caffe.io.resize( img, (SIZE, SIZE, 3) ) # resize to fixed size
     img = img.transpose(2,0,1)
     # you may apply other input transformations here...
     # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
     data_[i] = img
     for j in range(NUM_POINTS):
         #The coordinate values for each point are normalized
         if (j+1)%2:
             normalize_factor = width
         else:
             normalize_factor = height
         try:
             tmp = float(sp[j+1])/float(normalize_factor)
             label_[i][j] = tmp
         except ValueError,e:
             print "error", e, "on line", i
 
#with h5py.File('train.h5','w') as H:
with h5py.File('test.h5','w') as H:
    H.create_dataset( 'data', data=data_ ) # note the name X given to the dataset!
    H.create_dataset( 'label', data=label_ ) # note the name y given to the dataset!
with open('test_h5_list.txt','w') as L:
    L.write( 'train.h5' ) # list all h5 files you are going to use
