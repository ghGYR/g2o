import numpy as np


# anchor  x1 to z and x1=x2
def anchor(x1, x2, z, J=False):
    error = x1 - z
    if J:
        de_x1 = np.mat(np.eye(3))
        de_x2 = 0 * de_x1
        return error, de_x1, de_x2
    return error


# distance between x1 and x2
def distance(x1, x2, z, J=False):
    t = x2 - x1
    d = np.sqrt(t * t.T)
    # print(d)
    error = d - z
    if J:
        de_x1 = -t / d
        de_x2 = t / d
        return error, de_x1, de_x2
    return error


# transform from x1 to x2
def transform(x1, x2, z, J=True):
    t = x2 - x1
    error = t - z
    if J:
        de_x1 = -1 * np.mat(np.eye(3))
        de_x2 = -1 * de_x1
        return error, de_x1, de_x2
    return error


#  angle from x1 to x2
def angle(x1, x2, z, J=True):
    t = np.array(x2 - x1)
    w = t[0:2] / t[2]
    error = np.mat(w - z)
    if J:
        de_x1 = np.mat([
            [-1 / t[2], 0],
            [0, -1 / t[2]],
            [t[0] / t[2]**2, t[1] / t[2]**2]])
        de_x2 = -1 * de_x1
    return error, de_x1.T, de_x2.T
    return error
