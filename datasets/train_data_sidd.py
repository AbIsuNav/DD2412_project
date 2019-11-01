import cv2
import numpy as np
import glob
import os
import h5py as h5
"""
This class prepare the SIDD data. 
It crops the data into patches size 512 and saves them in a H5 file.

"""


def get_gt_images(datafolder):
    noisy_paths = sorted(glob.glob(os.path.join(datafolder, '**/*NOISY*.PNG'), recursive=True))
    gt_paths = [x.replace('NOISY', 'GT') for x in noisy_paths]
    return noisy_paths, gt_paths


def crop_image(im_folder="SIDD", path_file="SIDD_train.hdf5"):
    patch_size = 512
    stride = 512 - 128
    count = 0
    noisy_paths, gt_paths = get_gt_images(im_folder)
    with h5.File(path_file, 'w') as h5_file:
        for i in range(len(noisy_paths)):
            noisy_im = cv2.imread(noisy_paths[i])[:, :, ::-1]
            orig_im = cv2.imread(gt_paths[i])[:, :, ::-1]
            H, W, Ch = orig_im.shape
            crops_H = np.arange(0, H - patch_size + 1, stride)
            crops_W = np.arange(0, W - patch_size + 1, stride)
            for w_pt in crops_W:
                for h_pt in crops_H:
                    noisy_patch = noisy_im[h_pt:h_pt+patch_size, w_pt:w_pt+ patch_size, ]
                    gt_patch = orig_im[h_pt:h_pt+patch_size, w_pt:w_pt+ patch_size, ]
                    patches = np.concatenate((noisy_patch, gt_patch), axis=2)
                    h5_file.create_dataset(name=str(count), shape=patches.shape, dtype=patches.dtype, data=patches)
                    count += 1
    print("Finished data SIDD")


if __name__ == '__main__':
    crop_image()
