#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:32:10 2018

@author: Ruijie Ni
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Ultility functions

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

