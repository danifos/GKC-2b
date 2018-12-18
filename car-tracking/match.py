#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:46:03 2018

@author: Ruijie Ni
"""

# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import ultility


# %% Use track to match

class MeanShiftTracker:
    def __init__(self, img, pos, r=10):
        """
        Inputs:
            - img: First image to track
            - pos: Point to track
            - r: Size of track window
        """
        x, y = pos
        x, y = int(x), int(y)
        self.window = (x-r, y-r, 2*r, 2*r)
        roi = img[y-r:y+r+1, x-r:x+r+1]
        mask = cv.inRange(roi, np.array((0., 60.,32.)),
                          np.array((180.,255.,255.)))
        self.hist = cv.calcHist([roi], [0], mask, [180], [0,180])
        cv.normalize(self.hist, self.hist, 0, 255, cv.NORM_MINMAX)
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    def __call__(self, img):
        dst = cv.calcBackProject([img], [0], self.hist, [0,180], 1)
        _, self.window = cv.meanShift(dst, self.window, self.term_crit)
        x, y, w, h = self.window
        return x+w/2, y+h/2
    
    
class CornerTracker:
    def __init__(self, pos, r=10):
        """
        Inputs:
            - pos: Point to track
            - ~~threshold: Threshold for the white part~~
            - r: Size of track window
        """
        x, y = pos
        self.x, self.y = int(x), int(y)
        self.r = r
    def __call__(self, img, debug=False):
        W, H = img.shape[1], img.shape[0]
        cx = max(min(self.x, W-self.r-1), self.r)
        cy = max(min(self.y, H-self.r-1), self.r)
        
        if debug:
            ultility.show(img)
            
        roi = img[cy-self.r : cy+self.r+1, cx-self.r : cx+self.r+1]
        corner = cv.goodFeaturesToTrack(roi, 1, 0.5, 0)
        if corner is not None:
            corner = np.int16(corner[0]).reshape(-1) - self.r
            self.x = cx+corner[0]
            self.y = cy+corner[1]
        return self.x, self.y
    
Tracker = CornerTracker


# %% The interface

trackers = []

def match(img, debug=False):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    
    positions = []
    for tracker in trackers:
        positions.append(tracker(img, debug))
    positions = np.array(positions, dtype=np.float32)
    return positions


def init(img, positions):
    for pos in positions:
        trackers.append(Tracker(pos))
    

# %% Main: a test program

def main():
    cap = cv.VideoCapture('demo.mp4')
    ret, frame = cap.read(0)
    positions = np.array([[52.576714, 69.34139], [23.18219, 432.99365],
                          [634.5883,  426.2749], [585.03754, 56.74374]],
                         dtype=np.float32)  
    width = 297
    height = 210
    scale = 2
    width = int(width*scale)
    height = int(height*scale)
    perspective = np.array(((0,0), (0,height), (width,height), (width,0)),
                           dtype=np.float32)
    
    def warp_perspective(image):
        image = cv.warpPerspective(
            image,
            cv.getPerspectiveTransform(positions, perspective),
            (width, height)
        )
        return image
    
    img = warp_perspective(frame)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    init(frame, positions)
    while True:
        success, frame = cap.read(0)
        if not success:
            break
        
        plt.cla()
        positions = match(frame, False)
        ultility.show(frame)
        positions = np.concatenate([positions, positions[0:1,:]])
        plt.plot(positions[:,0], positions[:,1])
        plt.pause(0.03)


if __name__ == '__main__':
    main()
