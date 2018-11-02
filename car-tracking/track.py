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

import ultility

detector_type = 'SIFT'  # or 'SURF'


class Descriptor:
    def __init__(self, image, detector):
        self.img = image
        self.kp, self.des = detector.detectAndCompute(image, None)
    def show(self, item=None):
        points = np.array(tuple(p.pt for p in self.kp))
        ultility.show(self.img)
        #plt.scatter(points[:,0], points[:,1])
        if item:
            points = np.array(tuple(p.pt for p in self[item]))
            plt.scatter(points[:,0], points[:,1])
        #plt.show()
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.kp[item]
        if hasattr(item, '__getitem__') or isinstance(item, types.GeneratorType):
            return (self.kp[idx] for idx in item)
        return self.kp[item]


def regression(X_batch, Y_batch):
    print()
    N = X_batch.shape[0]
    learning_rate = 1e-6
    s, x, y, a = 1, 600, 800, np.pi
    indices = np.arange(N)
    for t in range(1000):
        stat = np.zeros(2)
        np.random.shuffle(indices)
        for i in range(N):
            X = X_batch[indices[i]]
            Y = Y_batch[indices[i]]
            
            atan = np.arctan2(X[1], X[0])
            norm = np.linalg.norm(X)
            sin = norm*np.sin(atan+a)
            cos = norm*np.cos(atan+a)
            Yhat = np.array((s*cos+x, s*sin+y))
            L = Y-Yhat
            stat += L
            
            ds = L*np.array((cos, sin))
            dx = L*np.array((1, 0))
            dy = L*np.array((0, 1))
            da = L*s*np.array((-sin, cos))
            
            s += learning_rate*np.sum(ds)
            x += learning_rate*500*np.sum(dx) - 5e-5*x
            y += learning_rate*500*np.sum(dy) - 5e-5*y
            a += learning_rate*np.sum(da)
            
            a %= np.pi*2
            
        if t%100 == 0:
            print(np.linalg.norm(stat))
    
    return s, x, y, a

def predict(x0, y0, s, x, y, a):
    atan = np.arctan2(y0, x0)
    norm = (x0**2+y0**2)**0.5
    return s*norm*np.cos(atan+a)+x, s*norm*np.sin(atan+a)+y

def main():
    if detector_type == 'SIFT':
        detector = cv.xfeatures2d.SIFT_create()
    elif detector_type == 'SURF':
        detector = cv.xfeatures2d.SURF_create()
    
    tmp = cv.imread('templates/car.jpg')
    tmp = cv.resize(tmp, (tmp.shape[1]//2, tmp.shape[0]//2))
    template = Descriptor(tmp, detector)
    image = Descriptor(cv.imread('IMG_0.jpg'), detector)
    
    bf = cv.BFMatcher()
    matches = bf.match(image.des, template.des)
    matches.sort(key=lambda x:x.distance)
    
    #template.show(m.trainIdx for m in matches[:16])
    #plt.show()
    ax = plt.figure().add_subplot(111)
    image.show(m.queryIdx for m in matches[:16])
    
    for m in matches[:16]:
        print(image[m.queryIdx].angle - template[m.trainIdx].angle)
    
    s, x, y, a = regression(
        np.array([template[m.trainIdx].pt for m in matches[:16]]),
        np.array([image[m.queryIdx].pt for m in matches[:16]])
    )
    
    print(s, x, y, a)
    
    x0, y0 = tuple(template[matches[0].trainIdx].pt)
    x1, y1 = tuple(image[matches[0].queryIdx].pt)
    print(x1, y1)
    print(predict(x0, y0, s, x, y, a))
    
    ax.add_patch(patches.Rectangle(
        predict(0, 0, s, x, y, a),
        template.img.shape[1]*s,
        template.img.shape[0]*s,
        a*180/np.pi,
        alpha=0.3
    ))
    plt.scatter(*predict(0, 0, s, x, y, a))
    plt.scatter(*predict(0, template.img.shape[0], s, x, y, a))
    plt.scatter(*predict(template.img.shape[1], 0, s, x, y, a))
    plt.scatter(*predict(template.img.shape[1], template.img.shape[0], s, x, y, a))
    

if __name__ == '__main__':
    main()