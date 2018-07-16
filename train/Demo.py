import numpy as np
import os
import cv2
import sys

_A = np.array  # A shortcut to creating arrays in command line 

sys.path.append('train')  # Assume git root directory

###########################    PATHS TO SET   ####################
# Either define CAFFE_ROOT in your enviroment variables or set it here
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','../caffe/distribute')
sys.path.append(CAFFE_ROOT+'/python')  # Add caffe python path so we can import it
sys.path.append("../caffe")
import caffe

#Import helper functions
from DataRow import DataRow, BBox, Predictor


PATH_TO_WEIGHTS  = os.path.join('/apps/liusj/snapshot/align', 'align_5p__iter_20000.caffemodel')
#PATH_TO_WEIGHTS  = os.path.join('/home/ly/workspace/Vanilla-40', 'caffeData', 'snapshots-40-unit-cvt1-inner', 'snap_iter_670000.caffemodel')
PATH_TO_DEPLOY_TXT = os.path.join('./train','landmark_deploy_5p.prototxt')
predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)

# Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
import dlib
detector=dlib.get_frontal_face_detector() # Load dlib's face detector


img_path = os.path.join('./data/images','000010.jpg')


img = cv2.imread(img_path)
#dets = detector( np.array(img, dtype = 'uint8' ), 1)
dets = [[1]]

for det in dets:

    det_box = BBox.BBoxFromLTRB(51, 238, 545, 656) #helen_3139620200_1

    print "det_box:  ",det_box.left, det_box.top, det_box.right, det_box.bottom

    #det_box = det_bbox
    #det_box.offset(-0.1*det_box.width(), 0.0*det_box.height())
    det_bbox = det_box


    scaledImage = DataRow()
    scaledImage.image = img
    scaledImage.fbbox = det_bbox
    dataRow60 = scaledImage.copyCroppedByBBox(det_bbox)
    image, lm = predictor.preprocess(dataRow60.image, dataRow60.landmarks())
    prediction = predictor.predict(image)
    #dataRow60.prediction = (prediction+0.5)*60.  # Scale -0.5..+0.5 to 0..60
    dataRow60.prediction = prediction*60.  # Scale -0.5..+0.5 to 0..60
    scaledImage.prediction = dataRow60.inverseScaleAndOffset(dataRow60.prediction) # Scale up to the original image scale
    ALIGNED_DATA_PATH = os.path.join('./', os.path.basename(img_path)[:-4]+'_reuslt.jpg')
    aligned_im = scaledImage.show()
    print ALIGNED_DATA_PATH
    print aligned_im.shape
    cv2.imwrite(ALIGNED_DATA_PATH, aligned_im)

