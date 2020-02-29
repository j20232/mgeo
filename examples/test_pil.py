import mgeo
import numpy as np
from PIL import Image
from mgeo import transform


if __name__ == "__main__":
    # show image with PIL
    img = Image.open("./assets/imgs/empire.jpg")
    img = transform.rotate(img, 0)
    img = transform.resize(img, (512, 512))
    points = np.array([[100, 200], [100, 500], [400, 200], [400, 500]])
    eq_img, cdf = transform.equalize_histogram(img[:, :, 0], nbr_bins=10)
    mgeo.utils.visualize.show_histogram(eq_img)
