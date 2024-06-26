# import torch

# class UNet(torch.nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(UNet, self).__init__()

#         self.enc_conv1 = self.conv_block(in_channels, 32)
#         self.enc_conv2 = self.conv_block(32, 64)
#         self.enc_conv3 = self.conv_block(64, 128)
#         self.enc_conv4 = self.conv_block(128, 256)

#         self.bottleneck = self.conv_block(256, 512)

#         self.dec_upconv4 = self.upconv_block(512, 256)
#         self.dec_conv4 = self.conv_block(512, 256)
#         self.dec_upconv3 = self.upconv_block(256, 128)
#         self.dec_conv3 = self.conv_block(256, 128)
#         self.dec_upconv2 = self.upconv_block(128, 64)
#         self.dec_conv2 = self.conv_block(128, 64)
#         self.dec_upconv1 = self.upconv_block(64, 32)
#         self.dec_conv1 = self.conv_block(64, 32)

#         self.final_conv = torch.nn.Conv2d(32, out_channels, kernel_size=1)

#     def forward(self, x):
#         enc1 = self.enc_conv1(x)
#         x = torch.nn.functional.max_pool2d(enc1, 2)
#         enc2 = self.enc_conv2(x)
#         x = torch.nn.functional.max_pool2d(enc2, 2)
#         enc3 = self.enc_conv3(x)
#         x = torch.nn.functional.max_pool2d(enc3, 2)
#         enc4 = self.enc_conv4(x)
#         x = torch.nn.functional.max_pool2d(enc4, 2)

#         x = self.bottleneck(x)

#         x = self.dec_upconv4(x)
#         x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=torch.nn.functionalalse)
#         x = x[:, :, :enc4.size(2), :enc4.size(3)]
#         x = torch.cat((x, enc4), dim=1)
#         x = self.dec_conv4(x)

#         x = self.dec_upconv3(x)
#         x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=torch.nn.functionalalse)
#         x = x[:, :, :enc3.size(2), :enc3.size(3)]
#         x = torch.cat((x, enc3), dim=1)
#         x = self.dec_conv3(x)

#         x = self.dec_upconv2(x)
#         x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=torch.nn.functionalalse)
#         x = x[:, :, :enc2.size(2), :enc2.size(3)]
#         x = torch.cat((x, enc2), dim=1)
#         x = self.dec_conv2(x)

#         x = self.dec_upconv1(x)
#         x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=torch.nn.functionalalse)
#         x = x[:, :, :enc1.size(2), :enc1.size(3)]
#         x = torch.cat((x, enc1), dim=1)
#         x = self.dec_conv1(x)

#         x = self.final_conv(x)
#         return x

#     def conv_block(self, in_channels, out_channels):
#         return torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             torch.nn.ReLU(inplace=True)
#         )

#     def upconv_block(self, in_channels, out_channels):
#         return torch.nn.Sequential(
#             torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             torch.nn.ReLU(inplace=True)
#         )

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        return logits
