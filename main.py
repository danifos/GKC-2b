#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:19:18 2018

@author: Ruijie Ni
"""

import sys, os
sys.path.append('car-tracking')
import process

# %% This is totally unrelated with any image

###############################################################################
# TODO:                                                                       #
#  Have anything you want to include. I think it's a good idea to make a new  #
#   directory in the root called "blue-tooth" or something, write scripts in  #
#   it and import them. You can also add some global variables.               #
###############################################################################
import geometry as ge
import serial
import numpy as np
import matplotlib.pyplot as plt

pos = 0
state = 0
check_time = 3
###############################################################################
#                              END OF THIS PART                               #
###############################################################################

ser = serial.Serial('/dev/rfcomm0')

def send(char):
    ser.write(str.encode(char))
    
def loop(path, car):
    """
    Receive position information and control the car.s
    
    Inputs:
        - path: List of tuples, which are the points of the path in order.
          [(x0, y0), (x1, y1), ...]
        - car: Tuple ((xt, yt), angle) of the current position and direction
          of the car.
    Returns:
        - state: A new dict including the information accumulated up to now.
    """
    ###########################################################################
    # TODO:                                                                   #
    #  Try to control the car in a smooth way. Please take possible errors    #
    #  into consideration. It's also POSSIBLE that `car` is None because the  #
    #  the car is not detected at one time. The parameter `state` may be      #
    #  handful, or you can use global variables and ignore it.                #
    ###########################################################################
    global state, pos
    
    if car[0] is None:
        print('Position not detected')
        return 0
    
    p_car = ge.Point(car[0])
    p_path = ge.Point(path[pos])
    line1 = ge.determine_linear_equation(p_car, p_path)
    if state == check_time:
        if p_car-p_path > 50:
            send("F")
        else:
            send("S")
            pos += 1
            if pos == len(path):
                return 1
            state = 0
            print('Turn to point {}'.format(pos))
    
    elif car[1]:
        delta = (np.arctan2(line1[1],line1[0])-np.pi/2)%(2*np.pi) - car[1]
        if delta < -np.pi: delta += np.pi*2
        elif delta > np.pi: delta -= np.pi*2
        
        if car[1] and abs(delta) > 0.3:
            if(delta < 0):
                send("L")
            else:
                send("R")
            state = 0
        else:
            send("S")
            state += 1
            print('Move to point {}'.format(pos))
    
    else:
        print('Angle not detected')
    
    return 0
    ###########################################################################
    #                            END OF THIS PART                             #
    ###########################################################################


# %% Main

def main():
    vertices = process.init()  # points of the path [(x0, y0), (x1, y1), ...]
    vertices = np.array(vertices)
    endpoints = [0, len(vertices)-1]
    
    def on_click(e):
        cur = np.array((e.xdata, e.ydata))
        dists = np.linalg.norm(vertices-cur, axis=1)
        p = np.argmin(dists)
        if e.button == 1:
            endpoints[0] = p
        elif e.button == 3:
            endpoints[1] = p
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    
    while True:
        ret = process.read(debug=True)  # info of the car ((xt, yt), angle)
        if ret == -1:
            print('Something wrong happened ...')
            os._exit(0)
        # do something to move the car
        points = None
        p = None
        start, end = endpoints
        if start < end:
            points = vertices[start:end+1]
            p = start+pos
        else:
            if end == 0:
                points = vertices[start::-1]
            else:
                points = vertices[start:end-1:-1]
            p = start-pos
        plt.scatter([vertices[p][0]], [vertices[p][1]], c='r')
        plt.scatter([vertices[end][0]], [vertices[end][1]], c='g')
        plt.pause(0.01)
        finish = loop(points, ret)
        if finish:
            print('The end')
            break
    ser.close()
        

if __name__ == '__main__':
    main()
