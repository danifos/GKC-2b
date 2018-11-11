#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:54:31 2018

@author: Ruijie Ni
"""

# %% The imports

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import extract
import localization
import ultility


# %% The constants

# about image collection
address = 'http://admin:9092@10.53.36.113:8081/video'
load_from_image = True
manual_transform = False
interval = 0.03

# sizes after transformation
width = 297
height = 210
scale = 2
width = int(width*scale)
height = int(height*scale)
perspective = np.array(((0,0), (0,height), (width,height), (width,0)),
                       dtype=np.float32)


# %% Functions for initialization

def init():
    if load_from_image:
        image = cv.imread('./board-images/image11-8.jpg')
    else:
        cap = cv.VideoCapture(address)
        if cap.isOpened():
            print('Connected to the IP camera sccessfully.')
        else:
            print('IP camera not available. Please check the settings.')
            os._exit(0)
           
        _, image = cap.read(0)
    
    plt.ion()
    fig = plt.figure()
    print('Original image:')
    ultility.show(image)
    plt.show()
    positions = []
    
    if manual_transform:
        def on_click(e):
            if len(positions) == 4: return
            if e.button == 1:
                positions.append((e.xdata, e.ydata))
                plt.scatter(e.xdata, e.ydata)
            plt.gca().figure.canvas.draw()
            
        fig.canvas.mpl_connect('button_press_event', on_click)
        while len(positions) < 4:
            plt.pause(interval)
    else:
        localization.init()
        tic = time.time()
        positions = localization.predict(image, debug=True)
        toc = time.time()
        print('Use time: {}'.format(toc-tic))
    
    positions = np.float32(np.array(positions))
    image = cv.warpPerspective(
        image,
        cv.getPerspectiveTransform(positions, perspective),
        (width, height)
    )
    
    vertices = extract.extract(image, debug=True)
    
    print('path:', vertices)
    ultility.show(image)
    ultility.plot(vertices, width, height)
    plt.show()
    
    return vertices


# %% The main process
        
def main():
    vertices = init()
    for t in range(0):
        cap = cv.VideoCapture(address)
        success, frame = cap.read(0)
        edges = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        plt.cla()
        plt.imshow(frame, edges)
        plt.pause(interval)


if __name__ == '__main__':
    main()
    
    