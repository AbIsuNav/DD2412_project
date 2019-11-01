#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage.measure import compare_ssim
import math
import torch
"""
tools for statistics used on training and validation of the model

"""


def calc_MSE(im_denoised,im_gt):
    mse = ((im_denoised-im_gt)**2).mean().item()
    return mse


def calculate_ssim(y, x, multichannel):
    ssim_avg = 0

    for i in range(y.shape[3]):
        ssim_avg += compare_ssim(y[:, :, :, i], x[:, :, :, i], win_size=11, data_range=255, multichannel=multichannel,gaussian_weights=True)

    return ssim_avg / (y.shape[3] + 1)


def psnr(img1, img2):
    mse = torch.mean((img1-img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))