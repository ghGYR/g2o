import numpy as np


class G2O():

    def __init__(self, graph):

        self.V, self.E, self.E_omega, self.E_model = graph

    def Lost(self):
        LSQ_error = 0
        for V1, V2, E_type, z in self.E:
            x1, x2, omega = self.V[V1], self.V[V2], self.E_omega[E_type]
            e = self.E_model[E_type](x1, x2, z)
            e_2 = e * omega * e.T
            LSQ_error += e_2 / 2
        return LSQ_error

    def Guass_Newton(self, lamda=0.001):
        num, dim = self.V.shape
        J = np.mat(np.zeros((num, dim)))
        H = np.zeros((num, dim, num, dim))
        # compute each edge J and H
        for V1, V2, E_type, z in self.E:
            x1, x2, omega = self.V[V1], self.V[V2], self.E_omega[E_type]
            e, je_x1, je_x2 = self.E_model[E_type](x1, x2, z, J=True)
            e_o = e * omega
            je1o = je_x1.T * omega
            je2o = je_x2.T * omega
            J[V1] += e_o * je_x1
            J[V2] += e_o * je_x2
            H[V1, :, V1, :] += je1o * je_x1
            # print(H[V1, :, V1, :])
            # input()
            H[V1, :, V2, :] += je1o * je_x2
            H[V2, :, V1, :] += je2o * je_x1
            H[V2, :, V2, :] += je2o * je_x2

        # set J,H to matrix
        J = J.reshape(J.size)
        H = np.mat(H.reshape(num * dim, num * dim))
        H = H + lamda * np.mat(np.eye(num * dim))
        # print(np.linalg.det(H))
        # update V
        delta_X = -1 * H.I * J.T
        # delta_X = -0.001 * J.T
        self.V += delta_X.reshape(num, dim)


if __name__ == "__main__":
    # sample
    from graph import *
    import mpl_toolkits.mplot3d
    import matplotlib.pyplot as plt
    test = G2O(test_graph)
    plt.ion()
    ax1 = plt.subplot(111, projection='3d')
    while True:

        ax1.clear()
        v_truth = np.array(V_truth).T
        v_pre = np.array(test.V).T
        ax1.scatter(v_pre[0], v_pre[1], v_pre[2], color="red", label="truth")
        ax1.scatter(v_truth[0], v_truth[1], v_truth[2], color="green", label="predction")
        ax1.legend()
        plt.pause(1)
        plt.show()
        test.Guass_Newton(lamda=0.0001)
        # print("V:", test.V)
        print("Lost:", test.Lost())
