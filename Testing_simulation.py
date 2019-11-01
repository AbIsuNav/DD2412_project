import torch
import generate_data as gd
import Networks.VDN as NET
import Networks.VDN_conf1 as NET1
from matplotlib import pyplot as plt
from pylab import *
from skimage.measure import compare_ssim
from skimage import img_as_ubyte, img_as_float
from tools import psnr
from torch.autograd import Variable
import os
"""
Testing of VDN simulation 

"""


def simulated_noise_test(noise_type, model_path, plot_noise=False, stats=False, plot_images=False, conf1=False):
    test_sets = {"LIVE1":"*.bmp", "CBSD68": "*.png", "Set5": "*.bmp"}
    test_paths = sorted(gd.load_data(os.path.join("datasets", "test_data"), test_sets))
    data_obj = gd.TestDataset(test_paths, noise_type=noise_type, iid=True)
    if torch.cuda.is_available():
        if conf1:
            model = NET1.VDN_NET(in_channels=3, depth_snet=5).cuda()
        else:
            model = NET.VDN_NET(in_channels=3, depth_snet=5).cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model = NET.VDN_NET(in_channels=3, depth_snet=5)
        model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()
    PSNR = 0.0
    SSIM = 0.0
    for idx in range(data_obj.__len__()):
        image, noisy = data_obj.__getitem__(idx)
        if torch.cuda.is_available():
            image, noisy = Variable(image.cuda()), Variable(noisy.cuda())
        _, ch, ht, wt = noisy.shape
        model_out, model_out2 = model(noisy)
        alpha = model_out2[:,:ch,]
        beta = model_out2[:,ch:,]
        result = beta/(alpha+1)
        noise = noisy - model_out[:, :ch, ]
        clean_img_pred = noise.view(ch, ht, wt).permute(1, 2, 0).clamp(0,1)
        image = image.view(ch, ht, wt).permute(1, 2, 0)
        if stats:
            PSNR += psnr(image*255, clean_img_pred*255)
            SSIM += compare_ssim(img_as_ubyte(image.cpu().detach().numpy()),img_as_ubyte(clean_img_pred.cpu().detach().numpy()),multichannel = True)
        if plot_noise:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            x = np.arange(0, ht, 1)
            y = np.arange(0, wt, 1)
            X, Y = np.meshgrid(x, y)
            Z1 = np.exp((img_as_float(result[0, 1, ].detach().numpy())))
            surf = ax.plot_surface(X[:100,:100], Y[:100,:100], Z1[:100,:100], rstride=1, cstride=1, cmap="viridis", antialiased=True, edgecolor="none")
            plt.savefig("simulation_results/approx_noise2_" + str(idx) + ".png")
            plt.show()
        if plot_images:
            plt.subplots_adjust(wspace=0.2)
            plt.subplot(131)
            plt.imshow(image[100:250,100:250,])
            plt.title('Groundtruth')
            plt.subplot(132)
            plt.imshow(noisy[0].permute(1, 2, 0)[100:250,100:250,])
            plt.title('Noisy Image')
            plt.subplot(133)
            plt.imshow(img_as_ubyte(clean_img_pred.detach().numpy())[100:250,100:250,])
            plt.title('Denoised Image')
            plt.savefig("simulation_results/denoised_simlation"+str(idx)+".png")
            plt.show()
    print("average PSNR = ", PSNR/data_obj.__len__())
    print("average SSIM = ", SSIM/data_obj.__len__())


if __name__ == '__main__':
    model_path = "trained_models/trained_model_sim_conf2.pth"
    simulated_noise_test(2,model_path, plot_noise=False, stats=False, plot_images=True, conf1=False)

