import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from mgeo.transform import warp

if __name__ == "__main__":
    img1 = np.array(Image.open("./assets/chap3/data/cat.jpg").convert("L"))
    img2 = np.array(Image.open(
        "./assets/chap3/data/billboard_for_rent.jpg").convert("L"))

    tp1 = np.array([[264, 538, 540, 264], [40, 36, 605, 605], [1, 1, 1, 1]])
    img3 = warp.image_in_image(img1, img2, tp1)

    tp2 = np.array([[675, 826, 826, 677], [55, 52, 281, 277], [1, 1, 1, 1]])
    img4 = warp.image_in_image(img1, img2, tp2)
    imlist = [img2, img3, img4]
    mgeo.utils.visualize.show_imgs(imlist)
