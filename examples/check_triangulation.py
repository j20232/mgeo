import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from mgeo.transform import Camera, sfm
from mpl_toolkits.mplot3d import axes3d


if __name__ == "__main__":
    im1 = np.array(Image.open('./assets/chap5/images/001.jpg'))
    im2 = np.array(Image.open('./assets/chap5/images/002.jpg'))
    points2D = [np.loadtxt('./assets/chap5/2D/00' +
                           str(i + 1) + '.corners').T for i in range(3)]
    points3D = np.loadtxt('./assets/chap5/3D/p3d').T

    # indices of correspondence points
    corr = np.genfromtxt('./assets/chap5/2D/nview-corners',
                         dtype='int', missing_values='*')

    P = [Camera(np.loadtxt('./assets/chap5/2D/00' + str(i + 1) + '.P'))
         for i in range(3)]

    X = np.vstack((points3D, np.ones(points3D.shape[1])))
    x = P[0].project(X)

    # --------------------- triangulation ------------------------
    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = np.vstack((x1, np.ones(x1.shape[1])))
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = np.vstack((x2, np.ones(x2.shape[1])))

    Xtrue = points3D[:, ndx]
    Xtrue = np.vstack((Xtrue, np.ones(Xtrue.shape[1])))
    Xest = sfm.triangulate(x1, x2, P[0].P, P[1].P)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.plot(Xtrue[0], Xtrue[1], Xtrue[2], 'r.')
    ax.plot(Xest[0], Xest[1], Xest[2], 'ko')
    plt.show()

    # ------------------- Project to 2D image ---------------------
    corr = corr[:, 0]
    ndx3D = np.where(corr >= 0)[0]
    ndx2D = corr[ndx3D]

    x = points2D[0][:, ndx2D]
    x = np.vstack((x, np.ones(x.shape[1])))
    X = points3D[:, ndx3D]
    X = np.vstack((X, np.ones(X.shape[1])))
    Pest = Camera(sfm.compute_P(x, X))
    xest = Pest.project(X)

    plt.figure(figsize=(8, 8))
    plt.imshow(im1)
    plt.plot(x[0], x[1], 'bo')
    plt.plot(xest[0], xest[1], 'r.')
    plt.axis('off')
    plt.show()
