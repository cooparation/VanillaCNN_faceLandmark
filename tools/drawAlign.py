#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math

input_path = "./data/try.txt"
output_folder = "output_tmp"
image_num = 0
for line in open(input_path):
    a = line.split()
    print "image", a[0]
    data = map(int, a[1:])
    pts = data[0:] # x1,y1,x2,y2...
    print pts
    image = cv2.imread(a[0])
    image_num += 1
    print "num point", len(data)
    for i in range(len(data)/2):
        cv2.circle(image, (pts[i*2], pts[i*2 + 1]), 2, (0,255,0), -1)
        cv2.putText(image, str(i), (pts[i*2], pts[i*2 + 1] -2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
    cv2.imwrite("%s/%d.jpg"%(output_folder, image_num), image)
