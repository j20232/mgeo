import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from mgeo.transform import homography

if __name__ == "__main__":
    img = np.array(Image.open("./assets/chap3/data/empire.jpg").convert("L"))
    H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
    img2 = ndimage.affine_transform(img, H[:2, :2], (H[0, 2], H[1, 2]))
    # mgeo.utils.visualize.show_imgs([img, img2])

    points = np.random.rand(3, 3)
    points = homography.normalize_in_homogeneous_coords(points)
