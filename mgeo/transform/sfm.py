import numpy as np
import matplotlib.pyplot as plt


def compute_fundamental(x1, x2):
    # x1, x2: (3, n)
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    A = np.zeros((n, 9))
    for i in range(n):
        # [x'x, x'y, x', y'x, y'y, y', x, y, 1]
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    return F


def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / (e[2] + 1e-6)


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    m, n = im.shape[:2]
    line = F @ x

    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    ndx = (lt >= 0) & (lt < m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if not show_epipole:
        return
    if epipole is None:
        epipole = compute_epipole(F)
    plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def triangulate_point(x1, x2, P1, P2):
    # triangulation with LSM via SVD
    M = np.zeros((6, 6))
    """
    [[P1 -x1  0 ]
     [P2   0 -x2]]
    """
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2
    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]
    return X / (X[3] + 1e-6)


def triangulate(x1, x2, P1, P2):
    # x1, x2: (3, n)
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return np.array(X).T


def compute_P(x, X):
    # x: 2d points (3, n), X: 3d points (4, n): homogeneous coords
    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # DLT
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = X[:, i]
        M[3 * i + 1, 4:8] = X[:, i]
        M[3 * i + 2, 8:12] = X[:, i]
        M[3 * i:3 * i + 3, i + 12] = -x[:, i]

    U, S, V = np.linalg.svd(M)
    return V[-1, :12].reshape((3, 4))
