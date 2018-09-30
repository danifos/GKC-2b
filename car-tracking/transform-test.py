#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 23:23:13 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('./IMG_0137.JPG').transpose((1,0,2))[::-1]
positions = []

def on_click(e):
    if positions == 4: return
    if e.button == 1:
        positions.append((e.xdata, e.ydata))
        plt.scatter(e.xdata, e.ydata)
    plt.gca().figure.canvas.draw()
    
plt.ion()
fig = plt.figure()
plt.imshow(image[:,:,::-1])
plt.show()
fig.canvas.mpl_connect('button_press_event',on_click)
while len(positions) < 4:
    plt.pause(0.003)
positions = np.float32(np.stack(positions, axis=0))
perspective = np.array(((0,0),(0,210),(297,210),(297,0)), dtype=np.float32)
image = cv.warpPerspective(image, cv.getPerspectiveTransform(positions, perspective), (291,210))
plt.imshow(image[:,:,::-1])
plt.show()