import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    img1 = np.array(Image.open(
        "./assets/chap2/data/sf_view1.jpg").convert("L"))
    img2 = np.array(Image.open(
        "./assets/chap2/data/sf_view2.jpg").convert("L"))
    sift_file1 = "./assets/chap2/data/sf_view1.jpg.sift"
    sift_file2 = "./assets/chap2/data/sf_view2.jpg.sift"

    sift = mgeo.feature.Sift()
    matcher = mgeo.feature.FeatureMatcher()
    l1, d1 = sift.read_features_from_file(sift_file1)
    l2, d2 = sift.read_features_from_file(sift_file2)

    matches = matcher.match_twosided(d1, d2)
    matcher.show_matches(img1, img2, l1, l2, matches)
