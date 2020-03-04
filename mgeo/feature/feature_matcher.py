import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class FeatureMatcher():

    def get_descriptors(self, img, points, wid=5):
        # local pixel values for each points
        return [img[p[1] - wid:p[1] + wid + 1,
                    p[0] - wid:p[0] + wid + 1].flatten()
                for p in points]

    def match(self, desc1, desc2, threshold=0.5):
        # select correspondence points with normalized cross-correlation
        distance = -np.ones((len(desc1), len(desc2)))
        n = len(desc1[0])
        print("matching...")
        for i in tqdm(range(len(desc1))):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            for j in range(len(desc2)):
                d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
                ncc_value = np.sum(d1 * d2) / (n - 1)
                if ncc_value > threshold:
                    distance[i, j] = ncc_value
        ndx = np.argsort(-distance)
        match_scores = ndx[:, 0]
        return match_scores

    def match_twosided(self, desc1, desc2, threshold=0.5):
        matches_12 = self.match(desc1, desc2, threshold)
        matches_21 = self.match(desc2, desc1, threshold)

        # delete asymmetry points
        ndx_12 = np.where(matches_12 >= 0)[0]
        for n in ndx_12:
            if matches_21[matches_12[n]] != n:
                matches_12[n] = -1
        return matches_12

    def show_matches(self, img1, img2, points1, points2, match_scores,
                     figsize=(15, 15), show_below=True):
        img3 = self.__append_images(img1, img2)
        if show_below:
            img3 = np.vstack((img3, img3))

        plt.figure(figsize=(15, 15))
        plt.gray()
        plt.imshow(img3)
        cols1 = img1.shape[1]
        for i, m in enumerate(match_scores):
            if m > 0:
                plt.plot([points1[i][0], points2[m][0] + cols1],
                         [points1[i][1], points2[m][1]], c=np.random.rand(3,))
        plt.axis("off")
        plt.show()

    def __append_images(self, img1, img2):
        rows1 = img1.shape[0]
        rows2 = img2.shape[0]

        # equalize shapes of two images
        if rows1 < rows2:
            img1 = np.concatenate(
                (img1, np.zeros((rows2 - rows1, img1.shape[1]))), axis=0)
        else:
            img2 = np.concatenate(
                (img2, np.zeros((rows1 - rows2, img2.shape[1]))), axis=0)

        return np.concatenate((img1, img2), axis=1)
