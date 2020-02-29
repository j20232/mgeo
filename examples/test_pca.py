import mgeo
from mgeo.linalg import pca
import numpy as np
from PIL import Image

if __name__ == "__main__":
    img_list = mgeo.utils.io.get_image_pathlist("./assets/imgs/a_thumbs")
    S, V, mean_X = pca(img_list)
    img_size = np.array(Image.open(img_list[0])).shape
    out_list = [mean_X.reshape(img_size[0], img_size[1])]
    for i in range(7):
        out_list.append(V[i].reshape(img_size[0], img_size[1]))
    mgeo.utils.visualize.show_imgs(out_list, rows=2)
