from . import DnCNN, UNet
import torch.nn as nn
"""
Class for VDN network 
(combination of U-Net and DnCNN)
"""


class VDN_NET(nn.Module):
    def __init__(self, in_channels, depth_snet=5):
        super(VDN_NET, self).__init__()
        d_net = UNet.UNet(in_channels=in_channels, out_channels=in_channels*2)
        s_net = DnCNN.DnCNN(DnCNN.make_Layers(depth_snet, out_channels=in_channels*2), image_channels=in_channels)
        self.DNet = self.init_kaiming(d_net)
        self.SNet = self.init_kaiming(s_net)

    def init_kaiming(self, net):
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
        return net

    def forward(self, x):  # train mode
        Z = self.DNet(x)
        sigma = self.SNet(x)
        return Z, sigma
