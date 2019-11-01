#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import h5py as h5
from scipy.io import loadmat


def validation_data_SIDD(path5="SIDD_validation.hdf5", dir="SIDD_validation"):

    path_h5 = os.path.join(dir, path5)
    if not os.path.exists(dir):
        os.mkdir(dir)
    noisy_mat = loadmat(os.path.join(dir, 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']
    gt_mat = loadmat(os.path.join(dir, 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
    num_img, num_block, _, _, _ = gt_mat.shape
    cont = 0
    with h5.File(path_h5, 'w') as h5_file:
        for ii in range(num_img):
            for jj in range(num_block):
                im_noisy = noisy_mat[ii, jj, ]
                im_gt = gt_mat[ii, jj, ]
                imgs = np.concatenate((im_noisy, im_gt), axis=2)
                h5_file.create_dataset(name=str(cont), shape=imgs.shape, dtype=imgs.dtype, data=imgs)
                cont += 1
    print('Finish saving validation data!\n')


if __name__ == '__main__':
    validation_data_SIDD()