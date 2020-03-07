import numpy as np


def equalize_histogram(gray_img, nbr_bins=256):
    img_hist, bins = np.histogram(gray_img.flatten(), nbr_bins, density=True)
    cdf = img_hist.cumsum()
    cdf = cdf / cdf[-1]   # normalization
    out = np.interp(gray_img.flatten(), bins[:-1], cdf)
    out = out.reshape(gray_img.shape)
    if gray_img.dtype == "uint8":
        out = (out * 255).astype(np.uint8)
        cdf = cdf * 255
    return out, cdf
