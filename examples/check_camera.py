import mgeo
from mgeo.transform import transform3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    # Load image
    points = np.loadtxt("./assets/chap4/3D/house.p3d").T
    points = np.vstack((points, np.ones(points.shape[1])))

    # Define projection matrix
    P = np.hstack((np.eye(3), np.array([[0], [0], [-10]])))

    # Project world coordinate to image coordinate
    camera = mgeo.transform.Camera(P)
    x = camera.project(points)

    # Define rotation matrix
    r = 0.05 * np.random.rand(3)
    rot = transform3d.create_rotation_matrix(r)

    plt.figure()
    num_move = 20
    for t in range(num_move):
        camera.P = np.dot(camera.P, rot)
        x = camera.project(points)
        plt.scatter(x[0], x[1], c=np.atleast_2d(np.random.rand(3, )))
    plt.show()

    K = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])
    tmp = transform3d.create_rotation_matrix([0, 0, 1])[:3, :3]
    Rt = np.hstack((tmp, np.array([[50], [40], [30]])))
    camera = mgeo.transform.Camera(K@Rt)
    K1, R1, t1 = camera.factorize()
    print("K: ", K)
    print("Rt: ", Rt)
    print("K1: ", K1)
    print("R1: ", K1)
    print("t1: ", t1)

    c = camera.calculate_center()
    print("c: ", c)
