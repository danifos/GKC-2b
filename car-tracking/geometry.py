#!/usr/bin/env python3
# -*- coding: utf-8 -*-

epsilon = 1e-8

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def to_tuple(self):
        return (self.x, self.y)
    
    def to_list(self):
        return [self.x, self.y]
        
    def __eq__(self, p):
        if self.x == p.x and self.y == p.y:
            return True
        return False
    
    def __ne__(self, p):
        if p and self.x == p.x and self.y == p.y:
            return False
        return True

    def __sub__(self, p):
        """Returns the distance between 2 points."""
        return ((self.x-p.x)**2 + (self.y-p.y)**2) ** 0.5


def determine_linear_equation(p1, p2):
    """Determin an equation for a line."""
    x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
    if x1 == x2:
        if y1 == y2:
            print("Error: determine_linear_equation(p1, p2) got two same points: ({}, {}).".format(x1, y1))
            return None
        A, B = 1, 0
    else:
        A, B = y1-y2, x2-x1
    C = -A*x1-B*y1
    return (A,B,C)


def f(l, p):
    """Substitute a point into a linear equation"""
    return l[0]*p.x + l[1]*p.y + l[2]


def get_distance(l, p):
    """Get the distance from a point to a line"""
    return abs(f(l, p)) / (l[0]**2+l[1]**2)**0.5


def get_intersection(l1, l2):
    """Get the intersection of 2 lines"""
    D = float(l1[0]*l2[1] - l2[0]*l1[1])
    if D == 0:
        print("Error: get_intersection(l1, l2) got two same line: {}x + {}y = {}".format(l1))
        return None
    Dx = l1[1]*l2[2] - l2[1]*l1[2]
    Dy = l1[2]*l2[0] - l2[2]*l1[0]
    return Point(Dx/D, Dy/D)


def get_perpendicular_line(l, p):
    """Get the perpendicular line of a point to a line"""
    A, B = l[1], -l[0]
    C = -A*p.x-B*p.y
    return (A,B,C)


def calculate_determinant(p1, p2, p3):
    """
    Calculate a rank-3 determinant:  
      |p1.x p1.y 1|  
      |p2.x p2.y 1|  
      |p3.x p3.y 1|  
    """
    return p1.x*(p2.y-p3.y) - p1.y*(p2.x-p3.x) + p2.x*p3.y - p3.x*p2.y