#!/usr/bin/env python
import torchvision as vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.SSIM import SSIM

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class MSA_block(nn.Module):
    def __init__(self,in_channels,out_channels,use_bn = True,**kwargs):
        super(MSA_block,self).__init__()
        branch_out = in_channels//2
        self.branch1x1 = BasicConv(in_channels, out_channels, use_bn=use_bn,
                                   kernel_size=1)

        self.branch3x3 = nn.Sequential(
            BasicConv(in_channels,branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels, use_bn=use_bn,
                      kernel_size=3, padding=1,dilation = 1),
        )
        self.branch5x5 = nn.Sequential(
            BasicConv(in_channels, branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels, use_bn=use_bn,
                      kernel_size=5, padding=4,dilation = 2),
        )
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels, use_bn=use_bn,
                      kernel_size=7, padding=9,dilation = 3),
        )


    def forward(self, x):

            branch1x1 = self.branch1x1(x)
            branch3x3 = self.branch3x3(x)
            branch5x5 = self.branch5x5(x)
            branch7x7 = self.branch7x7(x)
            branch_out_feature = branch3x3+branch5x5+branch7x7
            out = torch.cat([branch_out_feature,branch1x1],1)
            return out


class DecodingBlock(nn.Module):
    def __init__(self,low,high,out,**kwargs):
        super(DecodingBlock,self).__init__()
        lowin = low
        highin = high
        self.F1 = BasicConv(lowin,highin,use_bn=True,kernel_size = 1)
        self.F2 = nn.ConvTranspose2d(highin,highin,kernel_size=3,padding=1)
        self.F3 = BasicConv(2* highin ,out,use_bn=True,kernel_size = 1)
        self.upsample = UpsamleBlock(highin,highin)

    def forward(self,low_feature,high_feature):
        f1_out = self.F1(low_feature)
        up = self.upsample(high_feature)
        f3_in = torch.cat([f1_out,up],1)
        f3_out = self.F3(f3_in)

        return f3_out

class UpsamleBlock(nn.Module):
    def __init__(self,in_channels,out_channels ):
        super(UpsamleBlock, self).__init__()
        self.conv = BasicConv(in_channels,out_channels,kernel_size = 3,stride = 1,padding = 1)

    def forward(self, x):
        x = F.interpolate(x,[x.shape[2]*2,x.shape[3]*2],mode='nearest')
        x = self.conv(x)
        return x

class SDN(nn.Module):
    def __init__(self,gray = False,log = None,use_bn = True,training = True):
        super(SDN,self).__init__()
        if gray:
            in_channels = 1
        else:
            in_channels = 3
        if log is not None:
            self.log = log

        self.LE = nn.MSELoss(reduction='elementwise_mean')
        self.SSIM = SSIM(in_channel=3, window_size=11, size_average=True)
        dense = vision.models.densenet121(pretrained=True)
        backbone1 = nn.Sequential(dense.features[0:3])
        backbone2 = nn.Sequential(dense.features[4:6])


        self.Backbone1=backbone1
        self.Backbone2 =backbone2
        self.MSA1 = nn.Sequential(
            MSA_block(in_channels=128, out_channels=32),
            nn.MaxPool2d(2, 2)
        )

        self.MSA2 = nn.Sequential(
            MSA_block(in_channels=64, out_channels=16),
            nn.MaxPool2d(2, 2)
        )

        self.Decoding1 = DecodingBlock(low = 64,high = 32,out = 32)
        self.Decoding2 = DecodingBlock(low=128, high=32, out=16)
        self.Decoding3 = DecodingBlock(low=64, high=16, out=8)
        self.upsample1 = UpsamleBlock(8,8)



        self.FinalConv1 = BasicConv(in_channels=8,out_channels=4,use_bn=True,kernel_size=3,stride=1,padding=1)
        self.FinalConv2 = BasicConv(in_channels=4,out_channels=3,use_bn = True,kernel_size=1,stride=1)

    @property
    def loss(self):
        return self.loss_E + 0.001 * self.loss_C

    def  forward(self, x,gt_map):
        x1 = self.Backbone1(x)
        x2 = self.Backbone2(x1)
        x3 = self.MSA1(x2)
        x4 = self.MSA2(x3)

        up1 = self.Decoding1(x3,x4)
        up2 = self.Decoding2(x2,up1)
        up3 = self.Decoding3(x1,up2)
        up_final = self.upsample(up3)
        final1 = self.FinalConv1(up_final)
        es_map = self.FinalConv2(final1)

        if self.training:
            self.loss_E = self.LE(es_map, gt_map)
            self.loss_C = 1 - self.SSIM(es_map, gt_map)
        return es_map



