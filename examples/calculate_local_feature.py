import mgeo
from mgeo.transform import transform2d
import numpy as np
from PIL import Image


if __name__ == "__main__":
    # Load image
    img1 = np.array(Image.open("./assets/crans_1_small.jpg").convert("L"))
    img2 = np.array(Image.open("./assets/crans_2_small.jpg").convert("L"))
    img1 = transform2d.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
    img2 = transform2d.resize(img2, (img2.shape[1]//2, img2.shape[0]//2))

    # define detector
    harris = mgeo.feature.Harris()
    matcher = mgeo.feature.FeatureMatcher()

    # matching
    points1 = harris(img1)
    points2 = harris(img2)
    desc1 = matcher.get_descriptors(img1, points1)
    desc2 = matcher.get_descriptors(img2, points2)
    matches = matcher.match_twosided(desc1, desc2)
    matcher.show_matches(img1, img2, points1, points2, matches)
