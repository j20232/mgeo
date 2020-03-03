import mgeo
import numpy as np
from PIL import Image
from scipy.ndimage import filters

from mgeo import transform

if __name__ == "__main__":
    img = np.array(Image.open("./assets/empire.jpg").convert("L"))
    G = filters.gaussian_filter(img, 5)
    U, T = transform.transform2d.denoise(img, G)
    mgeo.utils.visualize.show_imgs([img, U, T])
