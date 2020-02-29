import numpy as np
from PIL import Image
import pathlib


def pca(X):
    if type(X) is list:
        if type(X[0]) == pathlib.PosixPath or type(X[0]) == str:
            tmp = np.array(
                [np.array(Image.open(im)).flatten() for im in X], "f")
        else:
            tmp = np.array([im.flatten() for im in X], "f")
        X = tmp
    # X: flattened data
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # High dimension
        M = X @ X.T  # Covariance
        eig_val, eig_vec = np.linalg.eigh(M)
        tmp = (X.T @ eig_vec).T
        V = tmp[::-1]  # Reverse components because tail vecs are important
        S = np.sqrt(eig_val)[::-1]
        V /= S[:, np.newaxis]
    else:
        # Low dimension: maybe same output of sklearn
        _, S, V = np.linalg.svd(X)
        V = V[:num_data]
    return S, V, mean_X  # Singular values, Components, Mean
