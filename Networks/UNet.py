import torch.nn as nn
import torch
import torch.nn.functional as F

"""
 UNet architecture
 
"""


class UNet(nn.Module):
    def encoder_block(self, in_channels, out_channels, kernel_size):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(out_channels),
            )
        return block

    def decoder_block(self, in_channels, out_channels, middle_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=middle_channels, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(middle_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=middle_channels, out_channels=middle_channels, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(middle_channels),
            nn.ConvTranspose2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=2,bias=True, stride=stride)
            )
        return block

    def final_layer(self, in_channel, middle_channel, out_channel, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channel, out_channels=middle_channel, padding=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(middle_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=middle_channel, out_channels=middle_channel, padding=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.BatchNorm2d(middle_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=middle_channel, out_channels=out_channel, padding=1,
                      bias=True),
        )
        return layer

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            diff = (bypass.size()[2] - upsampled.size()[2])
            c = diff // 2
            if diff % 2 != 0:
                bypass = F.pad(bypass, (-c, -c-1, -c, -c-1))
            else:
                bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def __init__(self, in_channels=1, out_channels=2, kernel_size=3):
        super(UNet, self).__init__()
        #ENCODER
        self.e1 = self.encoder_block(in_channels=in_channels, out_channels=64, kernel_size=kernel_size)
        self.encoder_avgpooling = nn.AvgPool2d(kernel_size=2)
        self.e2 = self.encoder_block(64, 128, kernel_size=kernel_size)
        self.e2_pooling = nn.AvgPool2d(kernel_size=2)
        self.e3 = self.encoder_block(128, 256, kernel_size=kernel_size)
        self.e3_pooling = nn.AvgPool2d(kernel_size=2)
        #BOTTLENECK
        self.bottleneck = self.decoder_block(in_channels=256, middle_channels=256, out_channels=256,
                                             kernel_size=kernel_size, stride=2, padding=1)
        #DECODER
        self.d1 = self.decoder_block(in_channels=512, middle_channels=256, out_channels=128, kernel_size=kernel_size,
                                     stride=2, padding=1)
        self.d2 = self.decoder_block(in_channels=256, middle_channels=128, out_channels=64, kernel_size=kernel_size,
                                     stride=2, padding=1)
        #LAST LAYER
        self.last_layer = self.final_layer(in_channel=128, middle_channel=64, out_channel=out_channels)

    def forward(self, x):
        #ENCODER
        encode1 = self.e1(x)
        #print("first encode:", encode1.shape)
        encode1_pool = self.encoder_avgpooling(encode1)
        #print("1st pool", encode1_pool.shape)
        encode2 = self.e2(encode1_pool)
        #print("second encode",encode2.shape)
        encode2_pool = self.e2_pooling(encode2)
        #print("2nd pool",encode2_pool.shape)
        encode3 = self.e3(encode2_pool)
        #print("third encode", encode3.shape)
        encode3_pool = self.e3_pooling(encode3)
        #print("3rd pool", encode3_pool.shape)
        #BOTTLENECK
        bottleneck = self.bottleneck(encode3_pool)
        #print("bottleneck (decode block): ", bottleneck.shape)

        #DECODER
        #print("decodeeer...")
        concat1 = self.crop_and_concat(bottleneck, encode3, crop=True)
        #print("first concat 1:", concat1.shape)
        decode1 = self.d1(concat1)
        #print("before second concat:", decode1.shape)
        concat2 = self.crop_and_concat(decode1, encode2, crop=True)
        #print("second concat:", concat2.shape)
        decode2 = self.d2(concat2)
        #print("second decode", decode2.shape)
        concat3 = self.crop_and_concat(decode2, encode1, crop=True)
        #print("3rd concanate:", concat3.shape)
        last_layer = self.last_layer(concat3)
        #print("the last layer: ", last_layer.shape)
        return last_layer



