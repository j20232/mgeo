import numpy as np
from scipy import linalg


class Camera():
    def __init__(self, P):
        self.P = P
        self.c = None

    def project(self, X):
        # project points X to normalized image coordinates
        x = np.dot(self.P, X)
        x /= x[2]
        return x

    def factorize(self):
        # factorize first 3*3 part. K: upper triangular, R: orthogonal
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = K@T
        self.R = T@R
        self.t = linalg.inv(self.K) @ self.P[:, 3]
        return self.K, self.R, self.t

    def calculate_center(self):
        if self.c is None:
            self.factorize()
            self.c = - self.R.T @ self.t
        return self.c
