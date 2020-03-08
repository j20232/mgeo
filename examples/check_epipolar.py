import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from mgeo.transform import Camera


if __name__ == "__main__":
    im1 = np.array(Image.open('./assets/chap5/images/001.jpg'))
    im2 = np.array(Image.open('./assets/chap5/images/002.jpg'))
    points2D = [np.loadtxt('./assets/chap5/2D/00' +
                           str(i + 1) + '.corners').T for i in range(3)]
    points3D = np.loadtxt('./assets/chap5/3D/p3d').T
    corr = np.genfromtxt('./assets/chap5/2D/nview-corners',
                         dtype='int', missing_values='*')
    P = [Camera(np.loadtxt('./assets/chap5/2D/00' + str(i + 1) + '.P'))
         for i in range(3)]

    X = np.vstack((points3D, np.ones(points3D.shape[1])))
    x = P[0].project(X)

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
