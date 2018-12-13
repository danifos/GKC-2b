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

###############################################################################
#                              END OF THIS PART                               #
###############################################################################

def loop(path, car, state):
    """
    Receive position information and control the car.s
    
    Inputs:
        - path: List of tuples, which are the points of the path in order.
          [(x0, y0), (x1, y1), ...]
        - car: Tuple (xt, yt) of the current position of the car.
        - state: Dict that may gives the information of the previou moments.
          It is initialized by an empty one {}.
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
    pass
    ###########################################################################
    #                            END OF THIS PART                             #
    ###########################################################################


# %% Main

def main():
    vertices = process.init()  # points of the path [(x0, y0), (x1, y1), ...]
    state = {}  # store some information
    while True:
        ret = process.read()  # coordinate of the car (xt, yt)
        if ret is None:
            print('Something wrong happened...')
            os._exit(0)
        # do something to move the car
        state = loop(vertices, ret, state)
        

if __name__ == '__main__':
    main()