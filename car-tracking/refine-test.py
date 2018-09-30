#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:01:24 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import geometry as gm
from point import Point

meta = [np.array([[1,1,1],[-1,0,0],[0,-1,0]]),
        np.array([[1,1,1],[0,0,0],[0,-1,-1]])]
templates = []
for t in meta:
    for i in range(4):
        templates.append(np.rot90(t, i))
        templates.append(np.rot90(t[:,::-1], i))
templates = np.stack(templates, axis=2)
    
def refine(image):
    # -1: must be 0; 1: must be 255; 0: either
    h, w = image.shape
    count = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i][j] == 255: continue
            roi = image[i-1:i+2, j-1:j+2].astype(np.int32)
            if np.sum(roi, axis=(0,1)) == 0: continue
            if np.min(np.sum((roi-1).reshape((3,3,-1)) * templates < 0, axis=(0,1))) == 0:
                image[i][j] = 255
                count += 1
    return count
    
plt.ion()
frame = cv.imread('./IMG_0136.JPG')
frame = cv.resize(frame, (240, 320))
edges = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
_, edges = cv.threshold(edges, 80, 255, cv.THRESH_BINARY)

s = 2
edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_CROSS, (s*2+1,)*2, (s,)*2))

while refine(edges): pass

#s = 1
#edges = cv.erode(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (s*2+1,)*2, (s,)*2))

plt.imshow(np.concatenate((frame[:,:,::-1], np.stack((edges,)*3, axis=2)), axis=1))
lines = cv.HoughLinesP(255-edges, 1, np.pi/180, 20, None, 30, 20)
lines = [l[0] for l in lines]
print(len(lines))

for line in lines:
    plt.plot(frame.shape[1]*2+line[::2], line[1::2])
    

# 后续操作
# 合并重合线段，连接断线
def remove(lst, obj):
    for idx, item in enumerate(lst):
        if item is obj:
            del lst[idx]
            return
    print(lst, obj)
    
delta = 3
flag = False
while not flag:
    flag = True
    for line1 in lines:
        if not flag: break
        for line2 in lines:
            if line1 is line2: continue
            p = [[Point(*line[:2]), Point(*line[2:])] for line in (line1, line2)]
            l = [gm.determine_linear_equation(*p[i]) for i in range(2)]
            if gm.get_distance(l[1], p[0][0]) <= delta and gm.get_distance(l[1], p[0][1]) <= delta:
                ends = p[0]
                pl = [gm.get_perpendicular_line(l[0], p[1][i]) for i in range(2)]
                out = [not ((gm.f(pl[i], p[0][0]) > 0) ^ (gm.f(pl[i], p[0][1]) > 0)) for i in range(2)]
                # 两外且同向，连接断线
                if out[0] and out[1] and not ((gm.f(pl[0], p[0][0]) > 0) ^ (gm.f(pl[1], p[0][0]) > 0)):
                    flag = False
                    for i in range(2):
                        if flag:
                            break
                        for j in range(2):
                            if ends[i]-p[1][j] <= 30:
                                ends[i] = p[1][1-j]
                                flag = True
                                break
                # 否则，合并重合线段
                else:
                    for i in range(2):
                        if out[i]:
                            ends[i] = p[1][i]
                remove(lines, line1)
                remove(lines, line2)
                lines.append(np.array(ends[0].to_tuple()+ends[1].to_tuple()))
                flag = False
                break

print(len(lines))
#for line in lines:
#    plt.plot(frame.shape[1]*3+line[::2], line[1::2])
    
# 将线段连成折线（已忽略孤立线段）
delta = 30
pinfo = []
linfo = []
for line in lines:
    linfo.append([Point(*line[:2]), Point(*line[2:])])
for idx1, line1 in enumerate(lines):
    for idx2, line2 in enumerate(lines):
        if idx2 <= idx1: continue
        flag = False
        p = [[Point(*line[:2]), Point(*line[2:])] for line in (line1, line2)]
        l = [gm.determine_linear_equation(*p[i]) for i in range(2)]
        for i in range(2):
            if flag:
                break
            for j in range(2):
                if p[0][i]-p[1][j] <= delta:
                    flag = True
                    _i, _j = i, j
                    break
        if flag:
            intersection = gm.get_intersection(*l)
            pinfo.append((intersection, idx1, idx2))
            linfo[idx1][_i] = intersection
            linfo[idx1][1], linfo[idx1][_i] = linfo[idx1][_i], linfo[idx1][1]
            linfo[idx2][_j] = intersection
            linfo[idx2][1], linfo[idx2][_j] = linfo[idx2][_j], linfo[idx2][1]

vertices = []

count = [0 for i in range(len(lines))]
for info in pinfo:
    count[info[1]] += 1
    count[info[2]] += 1
for i in range(len(count)):
    if count[i] == 1:
        cur = i
        break

vertices.append(linfo[cur][0].to_list())
flag = True
while pinfo and flag:
    flag = False
    for i, info in enumerate(pinfo):
        if info[1] == cur:
            cur = info[2]
        elif info[2] == cur:
            cur = info[1]
        else:
            continue
        vertices.append(info[0].to_list())
        del pinfo[i]
        flag = True
        break
vertices.append(linfo[cur][0].to_list())

vertices = np.array(vertices)
plt.plot(frame.shape[1]*3+vertices[:,0], vertices[:,1])
plt.scatter(frame.shape[1]*3+vertices[:,0], vertices[:,1])
plt.ylim((frame.shape[0]-1, 0))