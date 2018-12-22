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
import match


# %% The constants

# about image collection
username = 'admin'
password = 'admin'
lan = '10.134.20.153:8081'
address = None
load_from_video = False
manual_transform = False
fix_camera = False
interval = 0.03
frames_per_read = 5

# sizes after transformation
width = 297
height = 210
scale = 2
width = int(width*scale)
height = int(height*scale)
size = np.array((width, height))
positions = None
perspective = np.array(((0,0), (0,height), (width,height), (width,0)),
                       dtype=np.float32)
shift = 50  # show a larger perspective than the board

def warp_perspective(image, position, perspective, size):
    image = cv.warpPerspective(
        image, cv.getPerspectiveTransform(positions, perspective), tuple(size)
    )
    return image

# VideoCapture object
cap = None


# %% Functions for initialization

def init(debug=False):
    global lan, address, positions, cap, vertices
    
    track.init()
    if not manual_transform:
        localization.init()
    
    if load_from_video:
        cap = cv.VideoCapture('demo_large.mp4')
        #image = cv.imread('./board-images-new/image199.jpg')
    else:
#        string = input('Please check if the LAN address ({}) is correct \
#                       (press ENTER to confirm or type a new address):'\
#                       .format(lan))
#        if string != '':
#            lan = string
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
        tic = time.time()
        positions = localization.predict(image, debug=debug)
        toc = time.time()
        print('Use time: {}'.format(toc-tic))
    
    positions = np.float32(np.array(positions))
    print(positions)
    img = warp_perspective(image, positions, perspective, size)
    plt.cla()
    ultility.show(img)
    
    vertices = extract.extract(img, debug=debug)
    vertices = np.array(vertices)+shift
    print(vertices)
    
    if debug:
        print('path:', vertices)
        ultility.show(img)
        ultility.plot(vertices, width, height)
        plt.show()
    
    match.init(image, positions)
    if not load_from_video:
        cap = cv.VideoCapture(address)
    
    return vertices


# %% Read the current image and return the position of the car
    
def read(debug=False):
    global cap, positions
    success = -1
    tic = time.time()
    for t in range(frames_per_read):
        success, frame = cap.read(0)
    toc = time.time()
    print('Read a frame used {:.2f} s'.format(toc-tic))
    if not success: return -1
    
    if not fix_camera:
        positions = match.match(frame)
    img = warp_perspective(frame, positions, perspective+shift, size+2*shift)
    
    plt.cla()
    coords = track.track(img, debug=debug)
    if not debug:  # show the origin image if not in debug mode
        ultility.show(img, frame)
    ultility.plot(vertices)
    plt.pause(interval)
    
    return coords


# %% Test
        
def main():
    global load_from_video, manual_transform, fix_camera
    load_from_video = True
    manual_transform = False
    fix_camera = False
    
    init(debug=False)
    while True:
        plt.cla()
        ret = read(debug=True)
        if ret == -1:
            break
        plt.pause(interval)


if __name__ == '__main__':
    main()

