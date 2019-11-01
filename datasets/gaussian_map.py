import numpy as np

"""
Gaussian kernel use for simulation training noise
"""

def gen_gaussian_noise(patch_size, mu_x, mu_y, sigma):
    """
    creates gaussian kernel
    :param patch_size: int, size of a patch, final size of the mesh
    :param mu_x: int, center coordinate in x
    :param mu_y:  int, center coordinate in y
    :param sigma: int, variance
    :return: array with gaussian kernel

    """
    mu = np.array([mu_x, mu_y])
    Sigma = np.array([[sigma**2, 0.], [0., sigma**2]])
    X = np.linspace(0, patch_size, patch_size)
    Y = np.linspace(0, patch_size, patch_size)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    temp = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / temp


