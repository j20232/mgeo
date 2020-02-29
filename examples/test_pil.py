import mgeo
import numpy as np
from PIL import Image


if __name__ == "__main__":
    # show image with PIL
    pil_im = Image.open("./assets/imgs/empire.jpg")
    np_im = np.array(pil_im)
    mgeo.utils.visualize.show_thumbnail(np_im)
    image_path_list = mgeo.utils.io.get_image_pathlist("./assets/imgs")
