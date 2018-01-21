import numpy as np
from g2o.g2o import G2O
from rt3D.euler import *
from rt3D.graphic import*


# truth value
V_num = 10
L_num = 2
V_truth = np.mat(np.random.randint(V_num, size=(V_num, 3))) * 2
L_truth = np.mat([[4, 4, -4], [2, -2, -3]]) * 0.1  # add noise and rotation
R = rotation_x(0.1) * rotation_y(0.2) * rotation_z(0.2)
delta = 0.01
V_noise = V_truth + np.random.normal(loc=0, scale=delta, size=(V_num, 3))
V_nr = (R * V_noise.T).T
# draw_batch(np.array(V_nr), np.array(L_truth), t="truth", e="black", v="green")

# init values
R0 = rotation_x(0.82) * rotation_y(0.92) * rotation_z(1)
# R0 = np.mat(np.eye(3))
R0_I = R0.I
L0 = L_truth + np.random.normal(loc=0, scale=0.1, size=(L_num, 3))
# L0 = 0 * L0


# build g2o graph
gV = np.mat(np.zeros((1 + L_num, 9)))
gV[0] = R0_I.reshape(1, 9)
gV[1:L_num + 1, 0:3] = L0


def direction(R, l, z, J=False):
    V, s, = z
    a = R.reshape(3, 3) * V.T
    e1 = a[0, 0] - l[0, 0] - s[0] * (a[2, 0] - l[0, 2])
    e2 = a[1, 0] - l[0, 1] - s[1] * (a[2, 0] - l[0, 2])
    e = np.mat([e1, e2])
    # print("error", e)
    if J:
        de_R = np.mat(np.zeros((2, 9)))
        de_R[0, 0:3] = V
        de_R[1, 3:6] = V
        de_R[0, 6: 9] = -1 * s[0] * V
        de_R[1, 6: 9] = -1 * s[0] * V
        de_l = np.mat(np.zeros((2, 9)))
        de_l[0, 0] = -1
        de_l[1, 1] = -1
        de_l[0, 2] = s[0]
        de_l[1, 2] = s[1]
        return e, de_R, de_l
    return e


def rotation(R, c, z, J=False):
    r = R.reshape(3, 3)
    error = r * r.T - np.mat(np.eye(3))
    # print("e", r * r.T)
    error = error.reshape(1, 9)
    # print("R:",r)
    if J:
        de_R = np.mat(np.zeros((9, 9)))
        de_R[0, 0:3] = 2 * r[0]
        de_R[1, 0:3] = r[1]
        de_R[1, 3:6] = r[0]
        de_R[2, 0:3] = r[2]
        de_R[2, 6:9] = r[0]
        de_R[3, 0:3] = r[1]
        de_R[3, 3:6] = r[0]
        de_R[4, 3:6] = 2 * r[1]
        de_R[5, 3:6] = r[2]
        de_R[5, 6:9] = r[1]
        de_R[6, 0:3] = r[2]
        de_R[6, 6:9] = r[0]
        de_R[7, 3:6] = r[2]
        de_R[7, 6:9] = r[1]
        de_R[8, 6:9] = 2 * r[2]
        return error, de_R, np.mat(np.zeros((9, 9)))
    return error


E = np.array(np.empty((V_num * L_num + 1, 4), dtype=object))
# edge weight
belta = 0.8
E_omega = [(1 - belta) * np.mat(np.eye(2)), belta * np.mat(np.eye(9))]
E_model = [direction, rotation]
index = 0
for i in range(V_num):
    for j in range(L_num):
        t = V_truth[i] - L_truth[j]
        print(t)
        noise = np.random.normal(loc=0, scale=0.01, size=(1, 2))
        s1 = t[0, 0] / t[0, 2] + noise[0, 0]
        s2 = t[0, 1] / t[0, 2] + noise[0, 1]
        E[index, 0] = 0
        E[index, 1] = j + 1
        E[index, 2] = 0
        E[index, 3] = [V_nr[i], [s1, s2]]
        index += 1

E[-1, :] = [0, 0, 1, 0]
print(E)

graph = (gV, E, E_omega, E_model)
test = G2O(graph)
plt.ion()
while True:
    test.Guass_Newton(lamda=0.00001)
    R_I = test.V[0].reshape(3, 3)
    V_pre = R_I * V_nr.T
    L_pre = test.V[1:, 0:3]
    print(np.sum(np.array(V_pre.T - V_truth)**2))
    ax.clear()
    draw_batch(np.array(V_truth), np.array(L_truth), t="truth", e="red")
    # draw_batch(np.array(V_noise), np.array(L_truth), t="truth", e="yellow")
    draw_batch(np.array(V_pre.T), np.array(L_pre), t="pre", e="black", v="green")
    plt.pause(0.01)
    plt.show()
    # input()
