import numpy as np
import matplotlib.pyplot as plt


class Sift():

    def read_features_from_file(self, filename):
        f = np.loadtxt(filename)
        locs = f[:, :4]
        descs = f[:, 4:]
        return locs, descs

    def write_features_to_file(self, filename, locs, desc):
        np.savetxt(filename, np.hstack((locs, desc)))

    def plot_features(self, img, locs, figsize=(10, 10), circle=False):
        def draw_circle(c, r):
            t = np.arange(0, 1.01, .01) * 2 * pi
            x = r * np.cos(t) + c[0]
            y = r * np.sin(t) + c[1]
            plt.plot(x, y, 'b', linewidth=2)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        if circle:
            for p in locs:
                draw_circle(p[:2], p[2])
        else:
            plt.plot(locs[:, 0], locs[:, 1], 'ob')
        plt.axis('off')
        plt.show()
