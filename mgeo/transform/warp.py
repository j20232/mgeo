import numpy as np
from scipy import ndimage
from mgeo.transform import homography


def image_in_image(img1, img2, target):
    """
    Put img1 in img2 with an affine transformation
    such that corners are close to target as possible.

    Args:
        img1(np.ndarray): small image
        img2(np.ndarray): large image
        target(np.ndarray): homogeneous and counter-clockwise from top left
    """

    m, n = img1.shape[:2]
    source = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # calculate and apply affine transform: Please read scipy.ndimage.affine_transform
    H = homography.find_homography_with_HartleyZisserman(target, source)
    img1_t = ndimage.affine_transform(
        img1, H[:2, :2], (H[0, 2], H[1, 2]), img2.shape[:2])
    alpha = img1_t > 0
    return alpha * img1_t + (1 - alpha) * img2


def alpha_for_triangle(points, m, n):
    """ Creates alpha map of size (m,n)
        for a triangle with corners defined by points
        (given in normalized homogeneous coordinates). """

    alpha = np.zeros((m, n))

    for i in range(int(min(points[0])), int(max(points[0]))):
        for j in range(int(min(points[1])), int(max(points[1]))):
            x = np.linalg.solve(points, [i, j, 1])
            if min(x) > 0:  # all coefficients positive
                alpha[i, j] = 1
    return alpha
