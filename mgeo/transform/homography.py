import numpy as np
from PIL import Image
from scipy import linalg

from mgeo.transform.ransac import ransac


def rotate(img, angle=0):
    if type(img) is np.ndarray:
        img = Image.fromarray(np.uint8(img))
    return np.asarray(img.rotate(angle))


def create_rotation_matrix(a):
    # return a 3D rotation matrix for rotation around the axis of the vector a
    # p.72 at 3D rotation by [Kanatani. 2019]
    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]],
                             [a[2], 0, -a[0]],
                             [-a[1], a[0], 0]])
    return R


def convert_to_homogeneous_coords(points):
    return np.vstack((points, np.ones((1, points.shape[1]))))


def normalize_points(p):
    m = np.mean(p[:2], axis=1)
    maxstd = np.max(np.std(p[:2], axis=1)) + 1e-9
    C = np.diag([1 / maxstd, 1 / maxstd, 1])
    C[0][2] = -m[0] / maxstd
    C[1][2] = -m[1] / maxstd
    return C @ p, C


def find_homography_with_linearDLT(source, target):
    if source.shape != target.shape:
        raise RuntimeError("Number of points don't match!")
    source, C1 = normalize_points(source)
    target, C2 = normalize_points(target)

    nbr_correspondences = source.shape[1]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-source[0][i], -source[1][i], -1, 0, 0, 0,
                    target[0][i] * source[0][i], target[0][i] * source[1][i], target[0][i]]
        A[2 * i + 1] = [0, 0, 0, -source[0][i], -source[1][i], -1,
                        target[1][i] * source[0][i], target[1][i] * source[1][i], target[1][i]]
    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = np.linalg.inv(C2) @ H @ C1
    return H / H[2, 2]


def find_homography_with_HartleyZisserman(source, target):
    if source.shape != target.shape:
        raise RuntimeError("Number of points don't match!")
    source, C1 = normalize_points(source)
    target, C2 = normalize_points(target)

    # Multiple View Geometry in Computer Vision p.130
    A = np.concatenate((source[:2], target[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((C @ np.linalg.pinv(B), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    H = np.linalg.inv(C2) @ H @ C1
    return H / H[2, 2]

# ------------------------------ RANSAC-based ---------------------------------------


def normalize_in_homogeneous_coords(points):
    # normalize a collection of points in homogeneous coordinates
    for row in points:
        row /= points[-1] + 1e-6
    return points


class RansacModel():
    """ Class for testing homography fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""

    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        data = data.T  # transpose to fit H_from_points()
        fp = data[:3, :4]  # from points
        tp = data[3:, :4]  # target points
        return find_homography_with_linearDLT(fp, tp)

    def get_error(self, data, H):
        """ Apply homography to all correspondences,
            return error for each transformed point. """
        data = data.T
        fp = data[:3]
        tp = data[3:]
        fp_transformed = H @ fp
        fp_transformed = normalize_in_homogeneous_coords(fp_transformed)
        return np.sqrt(np.sum((tp - fp_transformed) ** 2, axis=0))


def find_homography_with_RANSAC(source, target, model, maxiter=1000, match_threshold=10):
    """ Robust estimation of homography H from point correspondences using RANSAC
        (ransac.py from http: // www.scipy.org/Cookbook/RANSAC).

        input: source, target: (3 * n arrays) points in hom. coordinates.
    """

    # group corresponding points
    data = np.vstack((source, target))

    # compute H and return
    H, ransac_data = ransac(data.T, model, 4, maxiter,
                            match_threshold, 10, return_all=True)
    return H, ransac_data['inliers']
