#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:46:03 2018

@author: Ruijie Ni
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import types

import geometry as gm
import ultility
from annotation import gamma_correction

detector_type = 'SIFT'  # 'SIFT' or 'SURF'
tmp_src = 'templates/car.jpg'
num_samples = 32
scale = 8
tmp_gamma = 1
frame_gamma = 1

class Descriptor:
    def __init__(self, image, detector):
        self.org_img = image.copy()
        self.img = image
        self.img = color_threshold(self.img, (0, 0, 255), 250)
        self.kp, self.des = detector.detectAndCompute(self.img, None)
    def show(self, item=None):
        ultility.show(self.img)
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


def color_threshold(img, color, threshold):
    d = np.sum(np.abs(img - np.array(color).reshape((1,1,3))), axis=2)
    return np.stack(((d < threshold) * (255-d),)*3, axis=2).astype(np.uint8)


def track(img):
    image = Descriptor(img, detector)
    
    h, w, _ = template.img.shape
    
    bf = cv.BFMatcher()
    matches = bf.match(image.des, template.des)
    matches.sort(key=lambda x:x.distance)
    
    best_matches = matches[:num_samples]
    
    image.show(m.queryIdx for m in best_matches)
    
    homography = cv.findHomography(np.array([template[m.trainIdx].pt for m in best_matches]),
                                   np.array([image[m.queryIdx].pt for m in best_matches]),
                                   cv.RANSAC, 3)[0]
    
    results = cv.perspectiveTransform(
            np.array([[template[m.trainIdx].pt for m in best_matches]]),
            homography)[0]
    plt.scatter(results[:,0], results[:,1], color='y', alpha=0.5)
    
    corners = cv.perspectiveTransform(np.array([[(0,0), (w,h), (0,h), (w,0)]],
                                               dtype='float32'), homography)[0]
    plt.scatter(corners[:,0], corners[:,1])
    plt.plot(corners[:2,0], corners[:2,1], 'k-')
    plt.plot(corners[2:,0], corners[2:,1], 'k-')
    
    corners = [gm.Point(c) for c in corners]
    center = gm.get_intersection(gm.determine_linear_equation(*corners[:2]),
                                 gm.determine_linear_equation(*corners[2:]))
    w = corners[0]-corners[3]
    baseline = gm.determine_linear_equation(*corners[::3])
    h = gm.get_distance(baseline, center)*2
    dirline = gm.get_perpendicular_line(baseline, center)
    a = np.arctan2(dirline[0], -dirline[1])
    
    ax.add_patch(patches.Rectangle(
        corners[3].to_tuple(),
        h, w, a*180/np.pi,
        color='w', alpha=0.3
    ))
    
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    

def main():
    global ax
    ax = plt.figure().add_subplot(111)
#    tmp = cv.imread('IMG_0590.JPG')
#    tmp = cv.resize(tmp, (tmp.shape[1]//4, tmp.shape[0]//4))
#    track(tmp)
#    return
    cap = cv.VideoCapture('VID_1.mp4')
    while True:
        for i in range(5):
            success, frame = cap.read(0)
        if not success:
            break
        #frame = gamma_correction(frame, frame_gamma).astype(np.uint8)
        frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        plt.cla()
        track(frame)
        plt.pause(0.03)


def init():
    global detector, template
    if detector_type == 'SIFT':
        detector = cv.xfeatures2d.SIFT_create()
    elif detector_type == 'SURF':
        detector = cv.xfeatures2d.SURF_create()
    
    tmp_img = cv.imread(tmp_src)
#    tmp_img = cv.resize(tmp_img, (tmp_img.shape[1]//scale, tmp_img.shape[0]//scale))
#    tmp_img = cv.GaussianBlur(tmp_img, (7, 7), 1., 1.)
#    tmp_img = gamma_correction(tmp_img, tmp_gamma).astype(np.uint8)
#    ultility.show(tmp_img)
#    plt.show()
    
    template = Descriptor(tmp_img, detector)


init()

if __name__ == '__main__':
    #pass
    main()