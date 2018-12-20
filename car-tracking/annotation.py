#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:44:22 2018

@author: Ruijie Ni
"""

# %% The imports

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from ultility import gamma_correction


# %% The constanst

# about sampling
videos = ['IMG_0{}.JPG'.format(i) for i in range(700, 901)]
sample_rate = 30
interval = 0.03
num_aug = 19

# image size
width = 640
height = 480
center = np.array([[width, height]])/2

# storage
video_dir = './train-images'
data_dir = './board-images-new'
anna_name = 'annotations'
file_name = 'image'


# %% Data augmentation

def transformation(image, prepers, offsets):
    prepers = np.float32(prepers)
    postpers = np.float32(prepers+offsets)
    result = cv.warpPerspective(
        image,
        cv.getPerspectiveTransform(prepers, postpers),
        (width, height)
    )
    return result


def data_augmentation(image, anna, new_annas):
    for i in range(num_aug):
        result = gamma_correction(image, np.exp(np.random.normal(scale=.3)))
        
        prepers = np.reshape(anna['coords'], (4,2))
        offsets = np.random.uniform(low=-np.sign(prepers - center)*4,
                                    high=np.sign(prepers - center)*20)
        result = transformation(result, prepers, offsets)
        result = result.astype(np.int32) \
               + np.random.normal(0, 2, (height, width, 3)).astype(np.int32)
        result = result.clip(0, 255).astype(np.uint8)
        
        file_dir = '{}-{}.jpg'.format(anna['file_name'].rstrip('.jpg'),
                                      chr(ord('a')+i))
        cv.imwrite(os.path.join(data_dir, file_dir), result)
        new_annas.append({'file_name' : file_dir,
                          'coords' : np.reshape(prepers+offsets, -1)})


# %% Annotating images from a video

def click_on(fig):
    positions = []
    points = []
    quitted = []
    
    def on_click(e):
        if len(positions) == 4: return
        if e.button == 1:
            positions.append((e.xdata, e.ydata))
            points.append(plt.scatter(e.xdata, e.ydata))
        elif e.button == 3:
            if len(positions) > 0:
                del positions[-1]
                points[-1].remove()
                del points[-1]
        elif e.button == 2:
            quitted.append(True)
        plt.gca().figure.canvas.draw()
        
    fig.canvas.mpl_connect('button_press_event', on_click)
    while len(positions) < 4:
        if quitted: return
        plt.pause(interval)
        
    plt.cla()
    
    return positions


def annotate(address, annotations, fig):
#    cap = cv.VideoCapture(address)
#    if not cap.isOpened():
#        print('Failed')
#        return
    
    while True:
#        for i in range(sample_rate):
#            ret, image = cap.read()
#            if not ret:
#                print('End')
#                return
        image = cv.imread(address)
        if image.shape[0] > image.shape[1]:
            image = np.rot90(image)
        image = cv.resize(image, (width, height))
        plt.imshow(image[:,:,::-1])
        
        positions = click_on(fig)
        if not positions:
            break
            
        file_dir = '{}{}.jpg'.format(file_name, len(annotations)//(num_aug+1))
        cv.imwrite(os.path.join(data_dir, file_dir), image)
        annotations.append({'file_name' : file_dir,
                            'coords' : np.reshape(np.array(positions), -1)})
        
        data_augmentation(image, annotations[-1], annotations)
        break


# %% Main process

def main():
    try:
        os.mkdir(data_dir)
        print('Created new directory')
    except:
        print('Writting on existed directory')
    
    annotations = []
    
    plt.ion()
    fig = plt.figure()
    
    for video in videos:
        annotate(os.path.join(video_dir, video), annotations, fig)
    
    with open(os.path.join(data_dir, anna_name), 'wb') as fo:
        pickle.dump(annotations, fo)


def reset():
    annotations = None
    with open(os.path.join(data_dir, anna_name), 'rb') as fi:
        annotations = pickle.load(fi)
        
    new_annotations = []
    for info in annotations:
        filename = info['file_name']
        if filename[-6] == '-' or \
           filename == 'image121.jpg':
            continue
        image = cv.imread(os.path.join(data_dir, filename))
        new_annotations.append(info)
        data_augmentation(image, new_annotations[-1], new_annotations)
        
    with open(os.path.join(data_dir, anna_name), 'wb') as fo:
        pickle.dump(new_annotations, fo)
    
    
if __name__ == '__main__':
    reset()
    #main()
