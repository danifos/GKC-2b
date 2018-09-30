#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 函数库
# 用于计算点、线段、射线、直线之间的各种关系
# 以及确定、代入直线方程、计算行列式等基础运算

from point import Point
import sys

epsilon = 1e-8

# 确定直线方程
def determine_linear_equation(p1, p2):
    x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
    if x1 == x2:
        if y1 == y2:
            sys.stderr.write("Error: determine_linear_equation(p1, p2) got two same points: (%s, %s).\n"\
                             %(x1, y1))
            return None
        A, B = 1, 0
    else:
        A, B = y1-y2, x2-x1
    C = -A*x1-B*y1
    return (A,B,C)

# 代入直线方程
def f(l, p):
    return l[0]*p.x + l[1]*p.y + l[2]

# 点到直线距离
def get_distance(l, p):
    return abs(f(l, p)) / (l[0]**2+l[1]**2)**0.5

# 解直线交点
def get_intersection(l1, l2):
    D = float(l1[0]*l2[1] - l2[0]*l1[1])
    if D == 0:
        sys.stderr.write("Error: get_intersection(l1, l2) git two same line: %sx + %sy = %s"\
                         %l1)
        return None
    Dx = l1[1]*l2[2] - l2[1]*l1[2]
    Dy = l1[2]*l2[0] - l2[2]*l1[0]
    return Point(Dx/D, Dy/D)

# 点到直线的垂线
def get_perpendicular_line(l, p):
    A, B = l[1], -l[0]
    C = -A*p.x-B*p.y
    return (A,B,C)

# 计算行列式
# |p1.x p1.y 1|
# |p2.x p2.y 1|
# |p3.x p3.y 1|
def calculate_determinant(p1, p2, p3):
    return p1.x*(p2.y-p3.y) - p1.y*(p2.x-p3.x) + p2.x*p3.y - p3.x*p2.y