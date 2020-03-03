import mgeo
import numpy as np
from PIL import Image


if __name__ == "__main__":
    img = np.array(Image.open("./assets/empire.jpg").convert("L"))
    harris = mgeo.feature.Harris()
    points = harris(img)
    mgeo.utils.visualize.show(img, points)
