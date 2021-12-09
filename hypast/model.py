import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    device = torch.device("cuda:0")

else:
    device = torch.device("cpu")
    
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)


class SegmentationModel(nn.Module):
    
    def make_conv_block(self, in_channels, out_channels, padding, kernel_size=3):
        layers = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False,
                     ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        ]

        return nn.Sequential(*layers)
    
    def make_last_conv_block(self, in_channels, out_channels, padding, kernel_size=3):
        layers = [
            nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                        ),
        ]

        return nn.Sequential(*layers)
      
    def make_tranpconv_block(self, in_channels, out_channels, kernel_size, stride):
        layers = [
            nn.ConvTranspose2d(in_channels=in_channels,
                        out_channels=out_channels,
                      kernel_size = kernel_size,
                               stride = stride,
                     ),           
                      
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        ]

        return nn.Sequential(*layers)
    
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        self.conv0 = self.make_conv_block(in_channels=3, out_channels=48, padding = 1, kernel_size = 3)
        self.base1 = nn.Sequential(*list(model.children())[2])[:6]
        self.base2 = nn.Sequential(*list(model.children())[2])[6:10]
        self.base3 = nn.Sequential(*list(model.children())[2])[10:16]
        self.base4 = nn.Sequential(*list(model.children())[2])[16:22]
        
        self.transp_conv1 = self.make_tranpconv_block(in_channels = 160, out_channels = 56, kernel_size=2, stride=2)
        self.conv1 = self.make_conv_block(in_channels = 112, out_channels = 112, padding = 1, kernel_size=3)
        self.conv2 = self.make_conv_block(in_channels = 112, out_channels = 32, padding = 1, kernel_size=3)
        
        self.transp_conv2 = self.make_tranpconv_block(in_channels = 32, out_channels = 32, kernel_size=2, stride=2)
        self.conv3 = self.make_conv_block(in_channels=64, out_channels=64, padding = 1, kernel_size=3)
        self.conv4 = self.make_conv_block(in_channels=64, out_channels=48, padding = 1, kernel_size=3)
        
        self.transp_conv3 = self.make_tranpconv_block(in_channels = 48, out_channels = 48, kernel_size=2, stride=2)
        self.conv5 = self.make_conv_block(in_channels=96, out_channels=96, padding = 1, kernel_size=3)
        self.conv6 = self.make_conv_block(in_channels=96, out_channels=48, padding = 1, kernel_size=3)
        
        self.last_conv = self.make_last_conv_block(in_channels=48, out_channels=2, kernel_size=3, padding = 1)
        
    def forward(self, x):
        
        x = self.conv0(x)
        xbase1 = self.base1(x)
        xbase2 = self.base2(xbase1)
        xbase3 = self.base3(xbase2)
        xbase4 = self.base4(xbase3)
        
        xup1 = self.transp_conv1(xbase4)
        xcat1 = torch.cat((xbase2, xup1),1)
        xconv1 = self.conv1(xcat1)
        xconv2 = self.conv2(xconv1)
        
        xup2 = self.transp_conv2(xconv2)
        xcat2 = torch.cat((xbase1, xup2),1)
        xconv3 = self.conv3(xcat2)
        xconv4 = self.conv4(xconv3)
        
        xup3 = self.transp_conv3(xconv4)
        xcat2 = torch.cat((x, xup3),1)
        xconv5 = self.conv5(xcat2)
        xconv6 = self.conv6(xconv5)
        
        
        return self.last_conv(xconv6)
        
        
