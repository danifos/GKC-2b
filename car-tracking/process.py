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
import track


# %% The constants

# about image collection
username = 'admin'
password = '9092'
lan = '10.192.217.220:8081'
address = None
load_from_video = False
manual_transform = True
fix_camera = True
interval = 0.03

# sizes after transformation
width = 297
height = 210
scale = 2
width = int(width*scale)
height = int(height*scale)
positions = None
perspective = np.array(((0,0), (0,height), (width,height), (width,0)),
                       dtype=np.float32)

def warp_perspective(image):
    image = cv.warpPerspective(
        image,
        cv.getPerspectiveTransform(positions, perspective),
        (width, height)
    )
    return image

# VideoCapture object
cap = None


# %% Functions for initialization

def init(debug=False):
    global lan, address, positions, cap
    
    if load_from_video:
        cap = cv.VideoCapture('demo.mp4')
        #image = cv.imread('./board-images-new/image199.jpg')
    else:
        string = input('Please check if the LAN address ({}) is correct \
                       (press ENTER to confirm or type a new address):'\
                       .format(lan))
        if string != '':
            lan = string
        address = 'http://{}:{}@{}/video'.format(username, password, lan)
        
        cap = cv.VideoCapture(address)
        if cap.isOpened():
            print('Connected to the IP camera sccessfully.')
        else:
            print('IP camera not available. Please check the settings.')
            os._exit(0)
           
    _, image = cap.read(0)
    
    if manual_transform or debug:
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
        positions = localization.predict(image, debug=debug)
        toc = time.time()
        print('Use time: {}'.format(toc-tic))
    
    positions = np.float32(np.array(positions))
    image = warp_perspective(image)
    
    vertices = extract.extract(image, debug=debug)
    
    if debug:
        print('path:', vertices)
        ultility.show(image)
        ultility.plot(vertices, width, height)
        plt.show()
    
    track.init()
    
    return vertices


# %% Read the current image and return the position of the car
    
def read(debug=False):
    global cap
    if not load_from_video:
        cap = cv.VideoCapture(address)
    success, frame = cap.read(0)
    if not success: return -1
    
    if fix_camera:
        img = warp_perspective(frame)
    else:  # TODO (maybe use SIFT to detect, or directly use CNN)
        pass
    
    plt.cla()
    coords = track.track(img, debug=debug)
    if not debug:  # show the origin image if not in debug mode
        ultility.show(frame)
    
    return coords


# %% Test
        
def main():
    global load_from_video, manual_transform
    load_from_video = True
    manual_transform = True
    
    path = init(debug=False)
    while True:
        plt.cla()
        ret = read(debug=True)
        if ret == -1:
            break
        plt.plot([p[0] for p in path], [p[1] for p in path])
        plt.pause(interval)


if __name__ == '__main__':
    main()

