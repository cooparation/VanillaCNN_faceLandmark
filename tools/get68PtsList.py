import os
import sys
import cv2
import math
from random import shuffle
from collections import OrderedDict

#paths = ['./data/images_68']
paths = ['/apps/liusj/FaceDatasets/300W/300W/01_Indoor', '/apps/liusj/FaceDatasets/300W/300W/02_Outdoor']
file_type_list =['GIF', 'gif', 'jpeg',  'bmp', 'png', 'JPG',  'jpg', 'JPEG']


write_lines = []
types = set()
num_images = 0
for path in paths:
    for root, _, files in os.walk(path):
        for fname in files:
            types.add(fname.split('.')[-1])
            if fname.split('.')[-1] in file_type_list:
                num_images += 1
                file_path = os.path.join(root,fname)
                label_path = os.path.join(root, fname.split('.')[-2]+".pts")
                lineNum = 0
                lineStr = str('')
                num_points = 0
                for line in open(label_path):
                    s = line.split()
                    if lineNum > 2 and lineNum < 71:
                        num_points += 1
                        x = str(int(math.floor(float(s[0]))))
                        y = str(int(math.floor(float(s[1]))))
                        lineStr += ' ' + x + ' ' + y
                    lineNum += 1
                if num_points != 68:
                    print file_path, 'num is', num_points
                    sys.exit()
                write_lines.append(file_path + lineStr + '\n')
#shuffle(write_lines)
print 'num images:', num_images
L  = int(len(write_lines))
f = open('data/image_68p.txt','w')
f.writelines(write_lines[:L])
f.close()
