import mgeo
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    img1 = np.array(Image.open(
        "./assets/chap4/book/book_frontal.JPG").convert("L"))
    img2 = np.array(Image.open(
        "./assets/chap4/book/book_perspective.JPG").convert("L"))
    sift_file1 = "./assets/chap4/book/book_frontal.sift"
    sift_file2 = "./assets/chap4/book/book_perspective.sift"

    sift = mgeo.feature.Sift()
    matcher = mgeo.feature.FeatureMatcher()
    l1, d1 = sift.read_features_from_file(sift_file1)
    l2, d2 = sift.read_features_from_file(sift_file2)

    # matches = matcher.match_twosided(d1, d2)
    # np.savetxt("./assets/chap4/book/matches.txt")
    matches = np.loadtxt("./assets/chap4/book/matches.txt")
    matches = matches.astype(np.uint8)

    matcher.show_matches(img1, img2, l1, l2, matches)
