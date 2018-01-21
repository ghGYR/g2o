import numpy as np
from math import sin, cos


def rotation_x(a):
    R = [[1, 0, 0], [0, cos(a), -1 * sin(a)], [0, sin(a), cos(a)]]
    return np.mat(R)


def rotation_y(a):
    R = [[cos(a), 0, -1 * sin(a)], [0, 1, 0], [sin(a), 0, cos(a)]]
    return np.mat(R)


def rotation_z(a):
    R = [[cos(a), sin(a), 0], [-1 * sin(a), cos(a), 0], [0, 0, 1]]
    return np.mat(R)


def rotation_xyz(a):
    pass
