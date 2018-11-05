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

detector_type = 'SIFT'  # 'SIFT' or 'SURF'
num_samples = 32


class Descriptor:
    def __init__(self, image, detector):
        self.img = image
        self.kp, self.des = detector.detectAndCompute(image, None)
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


def main():
    if detector_type == 'SIFT':
        detector = cv.xfeatures2d.SIFT_create()
    elif detector_type == 'SURF':
        detector = cv.xfeatures2d.SURF_create()
    
    template = Descriptor(cv.imread('templates/car.jpg'), detector)
    image = Descriptor(cv.imread('IMG_0.jpg'), detector)
    
    h, w, _ = template.img.shape
    
    bf = cv.BFMatcher()
    matches = bf.match(image.des, template.des)
    matches.sort(key=lambda x:x.distance)
    
    best_matches = matches[:num_samples]
    
    ax = plt.figure().add_subplot(111)
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
    

if __name__ == '__main__':
    main()