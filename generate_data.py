#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for loading and preprocess data
"""

import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import os
import datasets.gaussian_map as gm
import generate_simulated_noise as load_noise
import h5py as h5
from skimage import img_as_float32


def image_flip(image,k):
    if k == 0:
        return image
    elif k == 1:
        return np.rot90(image,k=1)
    elif k == 2:
        return np.rot90(image,k=2)
    elif k == 3:
        return np.rot90(image,k=3)
    elif k == 4:
        return np.flipud(image)
    elif k == 5:
        image = np.rot90(image,k=1)
        return np.flipud(image)
    elif k == 6:
        image = np.rot90(image,k=2)
        return np.flipud(image)
    elif k == 7:
        image = np.rot90(image,k=3)
        return np.flipud(image)


def generate_patches(im_gt, H, patch_size):
    patches = []
    for i in range(0,H-patch_size+1,10):
        for j in range(0,H-patch_size+1,10):
            patch = im_gt[i:i+patch_size,j:j+patch_size]
            patch = image_flip(patch,np.random.randint(0,8))
            patches.append(patch)
    
    return patches


def get_patch(image, patch_size):
    H,W,C = image.shape
    rW = random.randint(0, W-patch_size)
    rH = random.randint(0, H-patch_size)
    return image[rH:rH+patch_size, rW:rW+patch_size,:]


def get_patch_benchmark(image, patch_size):
    H,W,C = image.shape
    C = C//2
    rW = random.randint(0, W-patch_size)
    rH = random.randint(0, H-patch_size)
    noisy = np.array(image[rH:rH+patch_size, rW:rW+patch_size,:C])
    original = np.array(image[rH:rH+patch_size, rW:rW+patch_size,C:])
    return noisy, original


class TrainDataset(Dataset):
    """ class for dataloader"""

    def __init__(self, images_paths, batch_size=64,patch_size=128, channels=3, iid=0, randomize=True, extra_noise=True):
        # iid = 1 for iid , other value will be taken for not iid
        super(TrainDataset, self).__init__()
        self.data_paths = images_paths
        self.sigma = random.uniform(0, 75)  # CHECK THIS VALUE!
        self.patch_size = patch_size
        self.iid = True if iid == 1 else False
        self.randomize = randomize
        self.extra_noise = extra_noise
        self.sigma_max = 75
        self.channels = channels
        self.batch_size = batch_size

    def __len__(self):
        return 5000*self.batch_size

    def __getitem__(self, idx):
        """
        :param idx: index of the image to return
        :return: x = image gt
                 y = noisy image
                sigma_arr = sigma2_map_gt
        """
        idx = random.randint(0, len(self.data_paths)-1)
        image = cv2.imread(self.data_paths[idx], 1)
        if self.channels == 3:  # from BGR to RGB
            image = image[:, :, ::-1]
        x = get_patch(image, self.patch_size)
        x = np.multiply(x, (1 / 255))
        H, W, C = x.shape
        sigma_arr = self.noise_not_iid()
        y = x + np.random.randn(H, W, C) * sigma_arr
        sigma_arr = image_flip(sigma_arr, np.random.randint(0, 8))
        sigma_arr = np.tile(sigma_arr, (1, 1, self.channels))  # for training we always have 3 channels
        sigma_arr = np.where(sigma_arr < 1e-10, 1e-10, sigma_arr)  # took this from the code paper. It removes values
        # smaller than 1e-10
        sigma_arr = torch.tensor(sigma_arr)
        sigma_arr = sigma_arr.permute(2, 0, 1)

        x = image_flip(x, np.random.randint(0, 8)).copy()
        y = image_flip(y, np.random.randint(0, 8)).copy()
        y = torch.tensor(y)
        y = y.permute(2, 0, 1)
        x = torch.tensor(x).permute(2, 0, 1)
        return x.float(), y.float(), sigma_arr.float()

    def noise_iid(self):
        sigma_arr = np.ones((self.patch_size, self.patch_size))*(self.sigma / 255)
        return sigma_arr[:, :, np.newaxis]

    def noise_not_iid(self):
        if self.randomize:  # not explained in the paper, so we check this from paper code
            sigma = np.random.uniform(self.patch_size/4, self.patch_size/4*3)
            mu_x = np.random.uniform(0, self.patch_size)
            mu_y = np.random.uniform(0, self.patch_size)
        else:  # our method
            sigma = 50  # just a random number
            mu_x = self.patch_size//2
            mu_y = self.patch_size//2
        kernel = gm.gen_gaussian_noise(self.patch_size, mu_x, mu_y, sigma)
        if self.extra_noise:  # there is no clear explanation for this on the paper, so we check the code and include it
            self.sigma_max = 75  # from paper
            top_limit = random.uniform(0, self.sigma_max/255.0)
            low_limit = random.uniform(0, self.sigma_max/255.0)
            if top_limit < low_limit:
                top_limit, low_limit = low_limit, top_limit
            top_limit += 5/255.0  # from paper, no explanation for this number
            kernel_map = low_limit + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (top_limit - low_limit)
        else:  # our method
            kernel_map = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # normalization of noise
        return np.square(kernel_map[:, :, np.newaxis])


def load_data(data_path, train_sets):
    try:
        train_images_path = []
        for key, val in train_sets.items():
            train_images_path += [f for f in glob.glob(os.path.join(data_path, key, val), recursive=True)]
    except:
        print("Error in path given")
    return train_images_path


class TestDataset(Dataset):
    """
    This is the data load class for testing data with simulated noise.

    """
    def __init__(self, images_paths,noise_type=1, channels=3, iid=True):
        """

        :param images_paths: list with path of each image
        :param noise_type: int = 1,2,3
        :param channels: channels to load image, 1 or 3
        """
        super(TestDataset, self).__init__()
        self.data_paths = images_paths
        self.channels = channels
        self.noise_type = noise_type
        self.iid = iid

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        :param idx: index of the image to return
        :return: x = image gt
                 y = noisy image
        """
        if self.channels == 3:
            image = cv2.imread(self.data_paths[idx], 1)
            image = image[:, :, ::-1]  # from BGR to RGB
        elif self.channels == 1:
            image = cv2.imread(self.data_paths[idx], 0)
            image = np.expand_dims(image, axis=2)
        image = np.multiply(image, (1 / 255))  # normalize
        noise = load_noise.add_noise(image, self.noise_type, noise_iid=self.iid)[:, :, :self.channels]
        noisy = image + noise
        ht, wt, ch = noisy.shape
        ht = ht - int(ht % pow(2, 4))
        wt = wt - int(wt % pow(2, 4))
        noisy = noisy[:ht, :wt, :]
        image = image[:ht, :wt, :]
        noisy = noisy[np.newaxis, :, :, :]
        image = image[np.newaxis, :, :, :]
        y = torch.tensor(noisy)
        y = y.permute(0, 3, 1, 2)
        x = torch.tensor(image)
        x = x.permute(0, 3, 1, 2)
        return x.float(), y.float()


def sigma_benchmark(noisy, original, kernel, border):
    noise = (noisy - original)**2
    sigma_est = (cv2.GaussianBlur(noise,(kernel,kernel),border)).astype(np.float)
    sigma_est = np.where(sigma_est < 1e-10, 1e-10, sigma_est)
    return sigma_est


class TrainBenchmark(Dataset):
    """
    This is the data load class for testing data with simulated noise.

    """
    def __init__(self, h5_file_="datasets/SIDD_train.hdf5", patch_size=128, window=11, radius=5, batch_size=64):
        """

        :param h5_file_: path to file with patches saved from train_data_sidd.py
        :param patch_size: (int) size of patch used for training
        :param window: for Gaussian blur
        :param radius:  for Gaussian blur
        :return Y: noisy image
                X: original image
                Sigma_map: sigma map estimate
        """
        super(TrainBenchmark, self).__init__()
        with h5.File(h5_file_,'r') as h5_file:
            self.patch_num = list(h5_file.keys())
            self.total_images = len(self.patch_num)
        self.file5 = h5_file_
        self.patch_size = patch_size
        self.border = radius
        self.kernel_size = window
        self.batch_size = batch_size

    def __len__(self):
        return 5000*self.batch_size

    def __getitem__(self, index):
        indx = random.randint(0, self.total_images - 1)
        with h5.File(self.file5, 'r') as h5_file:
            image_patch = h5_file[self.patch_num[indx]]
            Y, X = get_patch_benchmark(image_patch, self.patch_size)
        X = img_as_float32(X)
        Y = img_as_float32(Y)
        X = image_flip(X, np.random.randint(0, 8))# data augmentation
        Y = image_flip(Y, np.random.randint(0, 8))
        sigma_map = sigma_benchmark(Y, X, self.kernel_size, self.border)
        Y = torch.tensor(Y.copy())
        Y = Y.permute(2, 0, 1)
        X = torch.tensor(X.copy())
        X = X.permute(2, 0, 1)
        sigma_map = torch.tensor(sigma_map)
        sigma_map = sigma_map.permute(2, 0, 1)

        return Y.float(), X.float(), sigma_map.float()


class ValidationBenchmark(Dataset):
    """
    This is the data load class for testing data with simulated noise.

    """
    def __init__(self, h5_file_="datasets/SIDD_validation/SIDD_validation.hdf5"):
        """

        :param h5_file_: path to file with patches saved from train_data_sidd.py
        :param patch_size: (int) size of patch used for training
        :param window: for Gaussian blur
        :param radius:  for Gaussian blur
        :return Y: noisy image
                X: original image
                Sigma_map: sigma map estimate
        """
        super(ValidationBenchmark, self).__init__()
        with h5.File(h5_file_,'r') as h5_file:
            self.patch_num = list(h5_file.keys())
            self.total_images = len(self.patch_num)
        self.file5 = h5_file_

    def __len__(self):
        return self.total_images

    def __getitem__(self, indx):
        with h5.File(self.file5, 'r') as file_5:
            images = file_5[self.patch_num[indx]]
            H, W, Ch = images.shape
            c = Ch//2
            Y = np.array(images[:, :, :c])
            X = np.array(images[:, :, c:])
        X = img_as_float32(X)
        Y = img_as_float32(Y)
        Y = torch.tensor(Y)
        Y = Y.permute(2, 0, 1)
        X = torch.tensor(X)
        X = X.permute(2, 0, 1)
        return Y, X


if __name__ == '__main__':
    # this is not working with the new stuff
    data_obj = TrainBenchmark()
    data = DataLoader(dataset=data_obj, drop_last=True, batch_size=5, shuffle=True)
    for i, batch_data in enumerate(data):
        x_batch, y_batch, sigma_arr = batch_data[1], batch_data[0], batch_data[2]
