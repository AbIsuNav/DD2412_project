import argparse

import torch
from pylab import *
from skimage import img_as_ubyte
from skimage.measure import compare_ssim

import Networks.VDN as NET
import generate_data as gd
from tools import psnr


def validation(model_path, file_data):
    if torch.cuda.is_available():
        model = NET.VDN_NET(in_channels=3, depth_snet=5).cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model = NET.VDN_NET(in_channels=3, depth_snet=5)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    avg_psnr_validation = 0.0
    avg_ssim_validation = 0.0
    obj_data = gd.ValidationBenchmark(h5_file_=file_data)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    for idx in range(obj_data.__len__()):
        noisy, image = obj_data.__getitem__(idx)
        ch, ht, wt = noisy.shape
        noisy = noisy.view(1, ch, ht, wt).cuda()
        image = image.cuda()
        model_out, _ = model(noisy)
        noise = noisy - model_out[:, :ch, ].detach().data
        clean_img_pred = noise.view(ch, ht, wt).permute(1, 2, 0).clamp(0, 1)
        image = image.view(ch, ht, wt).permute(1, 2, 0)
        avg_psnr_validation += psnr(image * 255, clean_img_pred * 255)
        avg_ssim_validation += compare_ssim(img_as_ubyte(image.cpu().numpy()),
                                            img_as_ubyte(clean_img_pred.cpu().numpy()),
                                            win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
    print("average validation PSNR = ", avg_psnr_validation / obj_data.__len__())
    print("average validation SSIM = ", avg_ssim_validation / obj_data.__len__())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to do validation of the benchmark data')
    parser.add_argument('--model', '-m', type=str, default="model.pth", help='path to trained model')
    parser.add_argument('--data', '-d', type=str, default="datasets/SIDD_validation/SIDD_validation.hdf5",
                        help='path to H5 file with validation data')
    arguments = parser.parse_args()
    validation(arguments.model, arguments.data)
