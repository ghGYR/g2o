import numpy as np
from edge import *

'''
a test graph  with  distance measured only
'''
v_num = 50
V_truth = np.mat(np.random.rand(v_num, 3)) * 100

# V0  init value for optimization
V0 = np.mat(V_truth + np.random.normal(loc=0, scale=10, size=(v_num, 3)))
E = []
for i in range(v_num):
    for j in range(i + 1, v_num):
        t = V_truth[i] - V_truth[j]
        d = np.mat(np.sqrt(t * t.T) + np.random.normal(loc=0, scale=0.1, size=1))
        d_edge = [i, j, 1, d]
        E.append(d_edge)

# fixed by 4 anchors
for i in range(4):
    E.append([i, i, 0, V_truth[i]])

# edges
E = np.array(E, dtype=object)
# remove some edge
num = len(E)
index = np.arange(num)[4:num - 4 - 1]
np.random.shuffle(index)
e_count = int(3 * (num - 8) // 10)
print("V number:%d\nEdge number:%d" % (v_num, e_count))
index = index[0:e_count]
index = list(index) + [0, 1, 2, 3, num - 1, num - 2, num - 3, num - 4]
E = E[index]

# information matrix
E_omega = [np.mat(np.eye(3)), np.mat([1])]
# edge model functions
E_model = [anchor, distance]
# test graph
test_graph = (V0, E, E_omega, E_model)
