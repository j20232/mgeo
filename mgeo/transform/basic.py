import numpy as np
from PIL import Image


def rotate(img, angle=0):
    if type(img) is np.ndarray:
        img = Image.fromarray(np.uint8(img))
    return np.asarray(img.rotate(angle))


def resize(img, size=(128, 128)):
    if type(img) is np.ndarray:
        img = Image.fromarray(np.uint8(img))
    return np.asarray(img.resize(size))
