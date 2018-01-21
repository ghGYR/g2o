import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

ax = plt.subplot(111, projection='3d')
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')


def draw_batch(V, L, t="graph", e="red", v="blue"):
    V, L = V.T, L.T
    V_num = V.shape[1]
    L_num = L.shape[1]
    for i in range(V_num):
        ax.text(V[0, i], V[1, i], V[2, i], "V%s" % (i + 1), color=v)
        ax.scatter(V[0, i], V[1, i], V[2, i], "V%s" % (i + 1), color=v)
        for j in range(V_num):
            ax.plot(V[0, [i, j]], V[1, [i, j]], V[2, [i, j]], color=e, linewidth=0.5)
        for j in range(L_num):
            ax.plot((V[0, i], L[0, j]), (V[1, i], L[1, j]), (V[2, i], L[2, j]), color=e, linewidth=0.25)
    for i in range(L_num):
        ax.text(L[0, i], L[1, i], L[2, i], "L%s" % (i + 1), color=v)
        ax.scatter(L[0, i], L[1, i], L[2, i], "V%s" % (i + 1), color=v)


