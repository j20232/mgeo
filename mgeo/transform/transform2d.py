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


def denoise(img, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """Rudin-Osher-Fatemi (ROF) model by A.Chambolle (2005)
    Reference: (11) of http://blanche.polytechnique.fr/preprint/repository/578.pdf

    Args:
        img (np.ndarray): noisy image
        U_init (np.ndarray): initial Gaussian distribution
        tolerance (float, optional): limits of error to finish iterations. Defaults to 0.1.
        tau (float, optional): step length. Defaults to 0.125.
        tv_weight (int, optional): weights of Total Variation(normalized). Defaults to 100.
    """
    m, n = img.shape

    # Initialization
    U = U_init
    Px = np.zeros((m, n))
    Py = np.zeros((m, n))
    error = 1

    while(error > tolerance):
        U_old = U

        # gradient of primal variable
        Grad_Ux = np.roll(U, -1, axis=1) - U  # x-component of U's gradient
        Grad_Uy = np.roll(U, -1, axis=0) - U  # y-component of U's gradient

        # update the dual variable
        Px_new = Px + (tau / tv_weight) * Grad_Ux
        Py_new = Py + (tau / tv_weight) * Grad_Uy
        norm_new = np.maximum(1, np.sqrt(Px_new ** 2 + Py_new ** 2))

        # normalization
        Px = Px_new / norm_new
        Py = Py_new / norm_new

        # update the primal variable
        rot_Px = np.roll(Px, 1, axis=1)  # right x rotation
        rot_Py = np.roll(Py, 1, axis=0)  # right y rotation

        div_p = (Px - rot_Px) + (Py - rot_Py)  # divergence of dual field
        U = img + tv_weight * div_p

        error = np.linalg.norm(U - U_old) / np.sqrt(n*m)
    return U, img - U
