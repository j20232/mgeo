import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mgeo.transform import homography, Camera
from mgeo.utils import gl


def my_calibration(sz):
    row, col = sz
    fx = 2555.0 * col / 2592
    fy = 2586.0 * row / 1936
    K = np.diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    return K


if __name__ == "__main__":
    img0 = np.array(Image.open(
        "./assets/chap4/book/book_frontal.JPG"))
    img1 = np.array(Image.open(
        "./assets/chap4/book/book_perspective.JPG"))
    sift_file0 = "./assets/chap4/book/book_frontal.sift"
    sift_file1 = "./assets/chap4/book/book_perspective.sift"
    box = mgeo.utils.visualize.cube_points([0, 0.0, 0.2], 0.1)
    width, height = 1000, 747

    sift = mgeo.feature.Sift()
    matcher = mgeo.feature.FeatureMatcher()
    l0, d0 = sift.read_features_from_file(sift_file0)
    l1, d1 = sift.read_features_from_file(sift_file1)
    matches = matcher.match_twosided(d0, d1)

    ndx = matches.nonzero()[0]
    fp = homography.convert_to_homogeneous_coords(l0[ndx, :2].T)
    ndx2 = [int(matches[i]) for i in ndx]
    tp = homography.convert_to_homogeneous_coords(l1[ndx2, :2].T)
    model = homography.RansacModel()
    H = homography.find_homography_with_RANSAC(fp, tp, model)[0]

    # Camera 1
    K = my_calibration((height, width))
    cam1 = Camera(np.hstack((K, K @ np.array([[0], [0], [-1]]))))
    box_cam1 = cam1.project(
        homography.convert_to_homogeneous_coords(box[:, :5]))
    box_trans = homography.normalize_in_homogeneous_coords(H @ box_cam1)

    # Camera 2
    cam2 = Camera(H @ cam1.P)
    A = np.linalg.inv(K) @ cam2.P[:, :3]
    A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
    cam2.P[:, :3] = K @ A
    box_cam2 = cam2.project(homography.convert_to_homogeneous_coords(box))

    plt.figure()
    plt.imshow(img0)
    plt.plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
    plt.axis('off')

    plt.figure()
    plt.imshow(img1)
    plt.plot(box_trans[0, :], box_trans[1, :], linewidth=3)
    plt.axis('off')

    plt.figure()
    plt.imshow(img1)
    plt.plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
    plt.axis('off')
    plt.show()
