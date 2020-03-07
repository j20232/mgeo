import numpy as np
import scipy
import scipy.linalg
import pylab

# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# * Neither the name of the Andrew D. Straw nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def ransac(data, model, n, k, t, d, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    Given:
        data - a set of observed data points
        model - a model that can be fitted to data points
        n - the minimum number of data values required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)
    """

    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """

    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = np.linalg.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = scipy.dot(A, model)
        # sum squared error per row
        err_per_point = np.sum((B-B_fit)**2, axis=1)
        return err_per_point


def test():
    # generate perfect input data
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = scipy.dot(A_exact, perfect_fit)
    assert B_exact.shape == (n_samples, n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    A_noisy = A_exact + numpy.random.normal(size=A_exact.shape)
    B_noisy = B_exact + numpy.random.normal(size=B_exact.shape)

    # add some outliers
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    non_outlier_idxs = all_idxs[n_outliers:]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)  # the first columns of the array

    # the last columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)]
    model = LinearLeastSquaresModel(input_columns, output_columns)
    linear_fit, resids, rank, s = np.linalg.lstsq(
        all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data, model,
                                     5, 5000, 7e4, 50,  # misc. parameters
                                     return_all=True)
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # maintain as rank-2 array
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    pylab.plot(A_noisy[ransac_data['inliers'], 0],
               B_noisy[ransac_data['inliers'], 0], 'bx', label='RANSAC data')
    pylab.plot(A_col0_sorted[:, 0], (A_col0_sorted @ ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0], (A_col0_sorted @ perfect_fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0], (A_col0_sorted @ linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    test()
