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

    # --------------------- visualize x ------------------------

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.plot(points2D[0][0], points2D[0][1], '*')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(im1)
    plt.plot(x[0], x[1], 'r.')
    plt.axis('off')

    plt.show()

    # --------------------- epipolar line ------------------------

    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = np.vstack((x1, np.ones(x1.shape[1])))
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = np.vstack((x2, np.ones(x2.shape[1])))

    F = sfm.compute_fundamental(x1, x2)
    e = sfm.compute_epipole(F)
    print(e)

    # plot epipolar line
    plt.figure(figsize=(8, 8))
    plt.imshow(im1)
    for i in range(5):
        sfm.plot_epipolar_line(im1, F, x2[:, i], e, False)
    plt.axis('off')
    plt.show()

    # --------------------- epipole ------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(im2)
    for i in range(5):
        plt.plot(x2[0, i], x2[1, i], 'o')
    plt.axis('off')
    plt.show()
