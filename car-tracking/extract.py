#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:57:08 2018

@author: Ruijie Ni
"""

# %% The imports

import numpy as np
import cv2 as cv
import geometry as gm
import matplotlib.pyplot as plt
import ultility

# %% The constants

# templates: 3-by-3 kernels used to check if a pixel can be deleted
# -1: must be 0; 1: must be 255; 0: either
meta = [np.array([[1,1,1],[-1,0,0],[0,-1,0]]),
        np.array([[1,1,1],[0,0,0],[0,-1,-1]])]
templates = []
for t in meta:
    for i in range(4):
        templates.append(np.rot90(t, i))
        templates.append(np.rot90(t[:,::-1], i))
templates = np.stack(templates, axis=2)

# s: When applying dialation, the size is (s*2+1, s*2+1), the kernel is (s, s).
s = 3

# distance that 2 lines closed enough to be recognized as a single one
delta_e = 10

# distance that 2 endpoints closed enough to be regonized as an intersection
delta_v = 30

margin = 1
use_adaptive_threshold = False
end_refine = float('inf')


# %% Ultility functions

def k_means_clustering(image):
    """
    Find 2 clusters of the colors of the image,
    and then return a boundary for the 2 clusters.
    
    Inputs:
    - image: An grayscale image.
    
    Returns:
    - threshold: The boundary that split the 2 clusters.
    """
    prev = np.zeros(2)
    centeroids = np.array([0., 255.])
    X = np.reshape(image, -1)
    sum_t = np.sum(X)
    while np.any(prev != centeroids):
        d = np.abs(X.reshape((1,-1)) - centeroids.reshape((-1,1)))
        indices = (d[0] != np.min(d, axis=0))
        num_clusters = np.sum(indices == np.array([[0],[1]]), axis=1)
        sum_1 = np.sum(X[indices])
        sum_clusters = np.array([sum_t-sum_1, sum_1])
        prev = centeroids.copy()
        centeroids = sum_clusters / num_clusters
    threshold = np.mean(centeroids)
    d = np.abs(X.reshape((1,-1)) - centeroids.reshape((-1,1)))
    clusters = (np.argmin(d, axis=0) == 1)
    return {'threshold': threshold,
            'center_min': centeroids[0],
            'center_max': centeroids[1],
            'max_min': np.max(X[clusters == 0]),
            'min_max': np.min(X[clusters == 1])}


def refine(image):
    """
    Extract the bone of a graph which is connected and thiner than before.  
    Specifically, the foreground is 0 and the background is 255.
    
    Inputs:
    - image: The image for input and output
    
    Returns:
    - count: Number of pixels deleted.
    """
    h, w = image.shape
    count = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i][j] == 255: continue
            roi = image[i-1:i+2, j-1:j+2].astype(np.int32)
            if np.sum(roi, axis=(0,1)) == 0: continue
            if np.min(np.sum(
                    (roi-1).reshape((3,3,-1)) * templates < 0,
                    axis=(0,1))) == 0:
                image[i][j] = 255
                count += 1
    return count


def remove(lst, obj):
    """
    Remove the same object the appears in a list.  
    Different from `list.remove(obj)`,
    this function use `is` instead of `==` as a condition.
    
    Inputs:
    - lst: The list
    - obj: The object
    """
    for idx, item in enumerate(lst):
        if item is obj:
            del lst[idx]
            return
        

def arrange(lines):
    """
    Delete overlapping lines, connect incontinuous lines.
    
    Inputs:
    - lines: List of lines to be rearranged.
      Each line is represent as a numpy array [P.x, P.y, Q.x, Q.y],
      where P and Q are the endpoints of the line.
    """
    flag = False
    while not flag:
        flag = True
        for line1 in lines:
            if not flag: break
            for line2 in lines:
                if line1 is line2: continue
                p = [[gm.Point(*line[:2]), gm.Point(*line[2:])] for line in (line1, line2)]
                l = [gm.determine_linear_equation(*p[i]) for i in range(2)]
                if gm.get_distance(l[1], p[0][0]) <= delta_e and gm.get_distance(l[1], p[0][1]) <= delta_e:
                    ends = p[0]
                    pl = [gm.get_perpendicular_line(l[0], p[1][i]) for i in range(2)]
                    out = [not ((gm.f(pl[i], p[0][0]) > 0) ^ (gm.f(pl[i], p[0][1]) > 0)) for i in range(2)]
                    if out[0] and out[1] and not ((gm.f(pl[0], p[0][0]) > 0) ^ (gm.f(pl[1], p[0][0]) > 0)):
                        flag = False
                        for i in range(2):
                            if flag:
                                break
                            for j in range(2):
                                if ends[i]-p[1][j] <= delta_v:
                                    ends[i] = p[1][1-j]
                                    flag = True
                                    break
                    else:
                        for i in range(2):
                            if out[i]:
                                ends[i] = p[1][i]
                    remove(lines, line1)
                    remove(lines, line2)
                    lines.append(np.array(ends[0].to_tuple()+ends[1].to_tuple()))
                    flag = False
                    break


def nodes(lines):
    """
    Find the intersections of a series of lines, as well as their orders.
    
    Inputs:
    - lines: List of the lines.
    
    Returns:
    - vertices: List of the nodes.
    """
    
    # detect all the intersections
    
    pinfo = []
    linfo = [[gm.Point(*line[:2]), gm.Point(*line[2:])] for line in lines]
    for info in linfo: info.append(gm.determine_linear_equation(*info))
    
    for idx1, line1 in enumerate(lines):
        for idx2, line2 in enumerate(lines):
            if idx2 <= idx1: continue
            flag = False
            p = [linfo[idx][:2] for idx in (idx1, idx2)]
            l = [linfo[idx][2] for idx in (idx1, idx2)]
            for i in range(2):
                if flag:
                    break
                for j in range(2):
                    if p[0][i]-p[1][j] <= delta_v:  # have an intersection!
                        flag = True
                        _i, _j = i, j
                        break
            if flag:
                intersection = gm.get_intersection(*l)
                pinfo.append((intersection, idx1, idx2))
                
                # always put the intersections in the second place,
                # so that the first place may be the endpoints
                linfo[idx1][_i] = intersection
                linfo[idx1][1], linfo[idx1][_i] = \
                linfo[idx1][_i], linfo[idx1][1]
                linfo[idx2][_j] = intersection
                linfo[idx2][1], linfo[idx2][_j] = \
                linfo[idx2][_j], linfo[idx2][1]
    
    # sort the intersections and the 2 endpoints
    
    vertices = []
    
    # which line has only one intersection (most have 2, some have 0, 2 have 1)
    count = [0 for i in range(len(lines))]
    for info in pinfo:
        count[info[1]] += 1
        count[info[2]] += 1
    for i in range(len(count)):
        if count[i] == 1:
            cur = i
            break
    
    vertices.append(linfo[cur][0].to_list())  # add the start
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
            vertices.append(info[0].to_list())  # add the next
            del pinfo[i]
            flag = True
            break
    vertices.append(linfo[cur][0].to_list())  # add the end
    
    return vertices


# %% Main process

def extract(frame, debug=False):
    """
    Extract the path in an image.
    
    Inputs:
    - frame: The input image.
    
    Returns:
    - vertices: The nodes of the path in order,
      though we don't care whether the first node is the start or the end.
      Each node is reporesented as a list [x, y]
    """
    
    # put the image from BGR to gray
    edges = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # ignore a boundary
    color = 0 if use_adaptive_threshold else 255
    edges[:margin,:] = color
    edges[-margin:,:] = color
    edges[:,:margin] = color
    edges[:,-margin:] = color
    # transform into binary image
    if use_adaptive_threshold:
        edges = cv.medianBlur(edges, 11)
        edges = cv.adaptiveThreshold(edges, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 41, 4)
    else:
        threshold = k_means_clustering(edges)['threshold']
        _, edges = cv.threshold(edges, threshold, 255, cv.THRESH_BINARY)
    
    # make the image thinner a little bit
    edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_CROSS, (s*2+1,)*2, (s,)*2))
    
    # make the image as thin as possible (until refine(edges) returns 0)
    #while refine(edges) > end_refine: pass

    # detect lines roughly with Hough transformation
    lines = cv.HoughLinesP(255-edges, 1, np.pi/360, 20, None, 30, 20)
    lines = [l[0] for l in lines]
    
    # show the binarization and Hough transform result for a while
    ultility.show(edges)
    for line in lines:
        plt.plot(line[::2], line[1::2])
    plt.pause(1)
    
    if debug:
        print('image refined | lines detected | lines rearranged')
        ultility.show(edges)
        for line in lines:
            plt.plot(frame.shape[1]+line[::2], line[1::2])
    
    # rearrange the lines and detect the nodes of the path.
    arrange(lines)
    
    if debug:
        for line in lines:
            plt.plot(frame.shape[1]*2+line[::2], line[1::2])
        plt.show()
            
    vertices = nodes(lines)
    
    # add the middle point of each segments
    new = []
    for i in range(len(vertices)-1):
        new.append(vertices[i])
        new.append(((vertices[i][0]+vertices[i+1][0])/2,
                    (vertices[i][1]+vertices[i+1][1])/2))
    new.append(vertices[-1])
    vertices = new
    
    # ensure that the start point is in the left
    if vertices[0][0] > vertices[-1][0]:
        vertices.reverse()
    
    return vertices
