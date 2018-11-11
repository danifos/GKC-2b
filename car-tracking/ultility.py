#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:32:10 2018

@author: Ruijie Ni
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Utility functions

def show(*images):
    arrays = [image[:,:,::-1] if len(image.shape) == 3 \
              else np.stack((image,)*3, axis=2) for image in images]
    plt.imshow(np.concatenate(arrays, axis=1))


def plot(vertices, width, height):
    vertices = np.array(vertices)
    plt.plot(vertices[:,0], vertices[:,1])
    plt.scatter(vertices[:,0], vertices[:,1])
    plt.xlim((0, width-1))
    plt.ylim((height-1, 0))


def channel_threshold(img, saturation, threshold, dim=1):
    d = np.abs(saturation - img[...,dim])
    img[...,1] *= (d < threshold)
    return img.astype(np.uint8)


def color_threshold(img, color, threshold):
    d = np.sum(np.abs(img - np.array(color).reshape((1,1,3))), axis=2, keepdims=True)
    img *= (d < threshold)
    return img.astype(np.uint8)
    #return np.stack(((d < threshold) * (255-d),)*3, axis=2).astype(np.uint8)


def gamma_correction(image, gamma):
    result = 256 * np.power(image/256, gamma)
    return result.astype(np.uint8)

