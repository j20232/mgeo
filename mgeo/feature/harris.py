import numpy as np
from scipy.ndimage import filters


class Harris():
    def __call__(self, img, sigma=3, min_dist=6, threshold=0.1):
        response = self.__calculate_response(img, sigma)
        return self.__get_points(response, min_dist, threshold)

    def __calculate_response(self, img, sigma=3):
        # derivatives
        img_x = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (0, 1), img_x)
        img_y = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (1, 0), img_y)

        # Calculate Harris matrix
        W_xx = filters.gaussian_filter(img_x * img_x, sigma)
        W_xy = filters.gaussian_filter(img_x * img_y, sigma)
        W_yy = filters.gaussian_filter(img_y * img_y, sigma)

        W_determinant = W_xx * W_yy - W_xy ** 2
        W_trace = W_xx + W_yy
        return W_determinant / W_trace

    def __get_points(self, res_img, min_dist=6, threshold=0.1):
        # min_dist: minimum pixel number to separate from corners

        # find corner candidates to exceed threshold
        corner_threshold = res_img.max() * threshold
        res_bool = (res_img > corner_threshold) * 1

        # get coordinates of candidates
        coords = np.array(res_bool.nonzero()).T
        candidate_values = [res_img[c[0], c[1]] for c in coords]
        candidate_indices = np.argsort(candidate_values)

        # points to seem as local descriptors
        allowed_locations = np.zeros(res_img.shape)
        allowed_locations[min_dist: -min_dist, min_dist:-min_dist] = 1

        filtered_coords = []
        for i in candidate_indices:
            if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
                filtered_coords.append(
                    [coords[i][1], coords[i][0]])  # yx -> xy
                allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
                                  (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0
        return np.array(filtered_coords)  # (number_of_points, 2)
