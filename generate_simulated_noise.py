from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from pylab import *
import numpy as np


def gen_test_noise1(x_size, y_size):
    """
    creates sincos kernel
    :param patch_size: int, size of a patch, final size of the mesh
    :return: array with gaussian kernel

    """
    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(0.1 * X) + np.cos(0.1 * Y)
    return Z


def gen_test_noise2(x_size, y_size):
    """
    creates ysin(x) - xcos(y) kernel
    :param patch_size: int, size of a patch, final size of the mesh
    :return: array with gaussian kernel

    """
    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    X, Y = np.meshgrid(x, y)
    Z = Y * np.sin(0.1 * X) - X * np.cos(0.1 * Y)
    return Z


def gen_test_noise3(x_size, y_size):
    """
    creates sin(sqrt(0.1(x^2+y^2))
    :param patch_size: int, size of a patch, final size of the mesh
    :return: array with gaussian kernel

    """
    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(0.1 * np.sqrt(X ** 2 + Y ** 2))
    return Z


def plot_noise(x_size, y_size, type=1):
    """
    Plot simulated noise
    :param x_size:
    :param y_size:
    :return:
    """
    # size = patch_size/2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, x_size, 1)
    y = np.arange(0, y_size, 1)
    X, Y = np.meshgrid(x, y)
    function_list = [gen_test_noise1(x_size, y_size), gen_test_noise2(x_size, y_size), gen_test_noise3(x_size, y_size)]
    if type == 1:
        noise = function_list[0]
    elif type == 2:
        noise = function_list[1]
    else:
        noise = function_list[2]
    noise = noise_min + (noise - noise.min()) / (noise.max() - noise.min()) * (noise_max - noise_min)
    surf = ax.plot_surface(X[:10, :10], Y[:10, :10], noise[:10, :10], rstride=1, cstride=1, antialiased=True)
    plt.show()


def gen_gaussian_wn(x_size, y_size, sigma):
    img = np.zeros((x_size, y_size))
    mean = 0.0
    std = sigma
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped


def add_noise(image, noise_type, noise_iid=True):
    noise_max = 75 / 255.0
    noise_min = 10 / 255.0
    ht, wt, ch = image.shape
    if noise_iid == True:
        function_list = [gen_test_noise1(wt, ht),
                         gen_test_noise2(wt, ht),
                         gen_test_noise3(wt, ht)]
        if noise_type == 1:
            noise = function_list[0]
        elif noise_type == 2:
            noise = function_list[1]
        else:
            noise = function_list[2]

    else:
        function_list = [gen_gaussian_wn(ht, wt, sigma=15),
                         gen_gaussian_wn(ht, wt, sigma=25),
                         gen_gaussian_wn(ht, wt, sigma=50)]
        if noise_type == 1:
            noise = function_list[0]
        elif noise_type == 2:
            noise = function_list[1]
        else:
            noise = function_list[2]
    noise = noise_min + (noise - noise.min()) / (noise.max() - noise.min()) * (noise_max - noise_min)
    temp_noise = np.random.randn(ht, wt, ch) * noise[:, :, np.newaxis]
    noise = np.concatenate((temp_noise, noise[:, :, np.newaxis]), axis=2)
    return noise


if __name__ == '__main__':
    noise_max = 75 / 255.0
    noise_min = 10 / 255.0
    plot_noise(512, 512)
