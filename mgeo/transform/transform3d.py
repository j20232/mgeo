import numpy as np
from scipy import linalg


def create_rotation_matrix(a):
    # return a 3D rotation matrix for rotation around the axis of the vector a
    # p.72 at 3D rotation by [Kanatani. 2019]
    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]],
                             [a[2], 0, -a[0]],
                             [-a[1], a[0], 0]])
    return R
