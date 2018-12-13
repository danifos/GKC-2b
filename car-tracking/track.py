#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:46:03 2018

@author: Ruijie Ni
"""

# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv

import ultility


# %% Use detection to track

class Descriptor:
    def __init__(self, correction):
        self.correction = correction
        self.success = False
        
    def __call__(self, image):
        self.org = image.copy()
        
        # Apply a same operation to get a recognizable feature
        # we've got 10 parameters here
        self.img = image
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        self.img = ((self.img[...,1:2] > 40) & (self.img[...,2:3] > 100)) \
                 * self.img
        self.front = (np.abs(self.img[...,0:1]-120) < 10) * self.img
        self.back = ((np.abs(self.img[...,0:1]-160) < 20) \
                   | (np.abs(self.img[...,0:1]-0) < 10)) * self.img
        self.back = self.refine(self.back, 2, 7)
        self.feat = cv.cvtColor(self.front + self.back, cv.COLOR_HSV2BGR)
        
        # Get 2 points from the image
        try:
            self.front_point = self.center_of_blob(self.front)
            self.back_point = self.center_of_blob(self.back)
            self.success = True
        except:
            self.success = False
            return None
        
        # Apply a correction of rotation
        dist = np.linalg.norm(self.back_point-self.front_point)
        ang = np.arctan2(self.back_point[1]-self.front_point[1],
                         self.back_point[0]-self.front_point[0]) \
            + self.correction
        self.back_point = self.front_point \
                        + dist * np.array((np.cos(ang), np.sin(ang)))
        
        return self.back_point, (ang+np.pi) % (np.pi*2)
        
    def show(self, item=None):
        ultility.show(self.org)
        if self.success:
            plt.gca().add_patch(patches.Arrow(
                *self.back_point, *(self.front_point-self.back_point),
                width=25, color='w'))
        
    def refine(self, img, s1, s2):
        img = cv.morphologyEx(img, cv.MORPH_OPEN,
            cv.getStructuringElement(cv.MORPH_CROSS, (s1*2+1,)*2, (s1,)*2))
        img = cv.dilate(img,
            cv.getStructuringElement(cv.MORPH_CROSS, (s2*2+1,)*2, (s2,)*2))
        return img
    
    def center_of_blob(self, img):
        _, img = cv.threshold(
            cv.cvtColor(cv.cvtColor(img, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY),
            1, 255, cv.THRESH_BINARY
        )
        M = cv.moments(img)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        return np.array((x, y))


def track(img, debug=False):
    """
    Inputs:
        - img: an image to be tracked
        - debug: to debug or not
    Returns a dict of:
        - center: numpy array (x, y), the center point of the car
        - angle: direction of the car, in rad
    """
    ret = detector(img)
    if debug: detector.show()
    return ret


def init():
    global detector
    detector = Descriptor(0.15)
    

# %% Main: a test program

def main():
    #cap = cv.VideoCapture('VID_4.mp4')
    address = 'http://admin:9092@10.64.15.42:8081/video'
    while True:
        cap = cv.VideoCapture(address)
        success, frame = cap.read(0)
        if not success:
            break
        # resize is only used in demo (the video is in a high resolution)
        #frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        plt.cla()
        #ultility.show(frame)
        track(frame, True)
        plt.pause(0.03)


if __name__ == '__main__':
    init()
    main()
