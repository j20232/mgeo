import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from mgeo.transform import warp

if __name__ == "__main__":
    img1 = np.array(Image.open("./assets/chap3/data/cat.jpg").convert("L"))
    img2 = np.array(Image.open(
        "./assets/chap3/data/blank_billboard.jpg").convert("L"))
    tp = np.array([[143, 353, 302, 50], [100, 30, 980, 922], [1, 1, 1, 1]])

    img3 = warp.image_in_image(img1, img2, tp)
    imlist = [img2, img3]
    mgeo.utils.visualize.show_imgs(imlist)
