# coding=utf-8
'''
Bob.Liu in 20171114
'''
import numpy as np
import math
import cv2

class LandmarkHelper(object):
    '''
    Helper for different landmark type
    '''
    @classmethod
    def parse(cls, line, landmark_type, has_bbox=False):
        '''
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input txt
            landmark_type: len of landmarks
        Return:
            see child parse
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            if has_bbox:
                return cls.__landmark5_bbox_txt_parse(line)
            else:
                return cls.__landmark5_nobbox_txt_parse(line)
        elif landmark_type == 68:
            return cls.__landmark68_txt_parse(line)
        elif landmark_type == 83:
            return cls.__landmark83_txt_parse(line)
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def flip(a, landmark_type):
        '''
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: len of landmarks
        Returns:
            landmarks: new landmarks
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            # left eye 0, right eye 1, nose, 2, left mouse 3, right mouse 4
            landmarks = np.concatenate((a[1,:], a[0,:], a[2,:], a[4,:], a[3,:]), axis=0)
        elif landmark_type == 68:
            landmarks = np.concatenate((a[16,:], a[15,:], a[14,:], a[13,:], a[12,:], a[11,:], a[10,:], a[9,:], a[8,:],
                a[7,:], a[6,:], a[5,:], a[4,:], a[3,:], a[2,:], a[1,:], a[0,:],
                a[26,:], a[25,:], a[24,:], a[23,:], a[22,:], a[21,:], a[20,:], a[19,:], a[18,:], a[17,:],
                a[27,:], a[28,:], a[29,:], a[30,:], a[35,:], a[34,:], a[33,:], a[32,:], a[31,:],
                a[45,:], a[44,:], a[43,:], a[42,:], a[47,:], a[46,:],
                a[39,:], a[38,:], a[37,:], a[36,:], a[41,:], a[40,:],
                a[54,:], a[53,:], a[52,:], a[51,:], a[50,:], a[49,:], a[48,:],
                a[59,:], a[58,:], a[57,:], a[56,:], a[55,:],
                a[64,:], a[63,:], a[62,:], a[61,:], a[60,:], a[67,:], a[66,:], a[65]), axis=0)
        elif landmark_type == 83:
            landmarks = np.concatenate((a[10:19][::-1], a[9:10], a[0:9][::-1], a[35:36],
                a[36:43][::-1], a[43:48][::-1], a[48:51][::-1], a[19:20], a[20:27][::-1],
                a[27:32][::-1], a[32:35][::-1], a[56:60][::-1], a[55:56], a[51:55][::-1],
                a[60:61], a[61:72][::-1], a[72:73], a[73:78][::-1], a[80:81], a[81:82],
                a[78:79], a[79:80], a[82:83]), axis=0)
        else:
            raise Exception("Unsupport landmark type...")
        return landmarks.reshape([-1, 2])

    @staticmethod
    def __landmark5_bbox_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=[0:4] is bbox and [4:] is landmarks
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        #data = map(int, a[1:])
        data = map(float, a[1:])
        pts = data[4:] # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark5_nobbox_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=[0:10] is landmarks
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        data = map(float, a[1:])
        pts = data[0:] # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark68_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks68
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        #a1 = np.fromstring(a[1], dtype=int, count=136, sep=',')
        #a1 = a1.reshape((-1, 2))
        #return a[0], a1
        data = map(float, a[1:])
        pts = data[0:]
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark83_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks83, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        a1 = np.fromstring(a[1], dtype=int, count=166, sep=',')
        a1 = a1.reshape((-1, 2))
        return a[0], a1

