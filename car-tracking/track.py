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
import types

import geometry as gm
import ultility
#from ultility import gamma_correction, channel_threshold, color_threshold


# %% consts

detector_type = 'SIFT'  # 'SIFT' or 'SURF'
tmp_src = 'templates/car.jpg'  # template image
num_samples = 32
# Only use and tweak these in emergency
#scale = 8
#tmp_gamma = 1.5
#frame_gamma = 0.7


# %% Use detection to track

class Descriptor:
    def __init__(self, image, detector):
        self.org = image.copy()
        
        # Apply a same operation to get a recognizable feature
        self.img = image
        #self.img = color_threshold(self.img, (0, 0, 255), 250)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        #self.img = channel_threshold(self.img, 195, 60, dim=0)
        
        self.kp, self.des = detector.detectAndCompute(self.img, None)
    def show(self, item=None):
        ultility.show(self.org)
        if item:
            points = np.array(tuple(p.pt for p in self[item]))
            plt.scatter(points[:,0], points[:,1])
        else:
            points = np.array(tuple(p.pt for p in self.kp))
            plt.scatter(points[:,0], points[:,1])
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.kp[item]
        if hasattr(item, '__getitem__') or isinstance(item, types.GeneratorType):
            return (self.kp[idx] for idx in item)
        return self.kp[item]


def track(img, debug=None):
    """
    Inputs:
        - img: an image to be tracked
        - debug: the subplot to draw mask if you want to debug
    Returns a dict of:
        - center: tuple (x, y), the center point of the car
        - angle: direction of the car, in rad
    """
    image = Descriptor(img, detector)
    
    h, w, _ = template.img.shape
    
    bf = cv.BFMatcher()
    matches = bf.match(image.des, template.des)
    matches.sort(key=lambda x:x.distance)
    
    best_matches = matches[:num_samples]
    
    homography = cv.findHomography(np.array([template[m.trainIdx].pt for m in best_matches]),
                                   np.array([image[m.queryIdx].pt for m in best_matches]),
                                   cv.RANSAC, 3)[0]
    
    results = cv.perspectiveTransform(
            np.array([[template[m.trainIdx].pt for m in best_matches]]),
            homography)[0]
    
    cpoints = cv.perspectiveTransform(np.array([[(0,0), (w,h), (0,h), (w,0)]],
                                               dtype='float32'), homography)[0]
    
    corners = [gm.Point(c) for c in cpoints]
    center = gm.get_intersection(gm.determine_linear_equation(*corners[:2]),
                                 gm.determine_linear_equation(*corners[2:]))
    w = corners[0]-corners[3]
    baseline = gm.determine_linear_equation(*corners[::3])
    h = gm.get_distance(baseline, center)*2
    dirline = gm.get_perpendicular_line(baseline, center)
    a = np.arctan2(dirline[0], -dirline[1])
    
    if debug:
        image.show(m.queryIdx for m in best_matches)
        
        plt.scatter(cpoints[:,0], cpoints[:,1])
        plt.plot(cpoints[:2,0], cpoints[:2,1], 'k-')
        plt.plot(cpoints[2:,0], cpoints[2:,1], 'k-')
        
        plt.scatter(results[:,0], results[:,1], color='y', alpha=0.5)
        
        debug.add_patch(patches.Rectangle(
            corners[3].to_tuple(),
            h, w, a*180/np.pi,
            color='w', alpha=0.3
        ))
        
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
    
    return {'center' : center.to_tuple(), 'angle' : a}


def init():
    global detector, template
    if detector_type == 'SIFT':
        detector = cv.xfeatures2d.SIFT_create()
    elif detector_type == 'SURF':
        detector = cv.xfeatures2d.SURF_create()
    
    template = Descriptor(cv.imread(tmp_src), detector)
    #ultility.show(cv.cvtColor(template.img, cv.COLOR_HSV2BGR))
    

# %% Main: a test program

def main():
    global ax
    ax = plt.figure().add_subplot(111)
#    tmp = cv.imread('IMG_0590.JPG')
#    tmp = cv.resize(tmp, (tmp.shape[1]//4, tmp.shape[0]//4))
#    track(tmp)
#    return
    cap = cv.VideoCapture('VID_1.mp4')
    while True:
        success, frame = cap.read(0)
        if not success:
            break
        # frame = gamma_correction(frame, frame_gamma)  # not to use it as possible (slow)
        # resize is only used in demo (the video is in a high resolution)
        frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        plt.cla()
        track(frame, ax)
        plt.pause(0.03)


if __name__ == '__main__':
    init()
    main()
