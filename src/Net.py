import torch
import torchvision
import torch.nn as nn
from src.SSIM import SSIM
from src.ssim_loss import SSIM_Loss
from matplotlib import pyplot as plt


class SANet(nn.Module):
    def __init__(self,gray = True):
        super(SANet, self).__init__()
        self.LE = nn.MSELoss(reduction='elementwise_mean')
        self.SSIM = SSIM(in_channel=3 , window_size=11,size_average=True)
        self.ssim_loss = SSIM_Loss(in_channels =3)

        if gray:
            in_channel = 1
        else:
            in_channel = 3

        self.FME = nn.Sequential(
            SANetModule(in_channel,16,1),
            nn.MaxPool2d(2,2),
            SANetModule(64,32,2),
            nn.MaxPool2d(2,2),
            SANetModule(128,32,3),
            nn.MaxPool2d(2,2),
            SANetModule(128,16,4)
        )

        self.DME = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=9,stride=1,padding=4),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(64,64,kernel_size=2,stride=2),
            nn.ReLU(),

            nn.Conv2d(64,32,kernel_size=7,stride=1,padding=3),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(32,32,kernel_size=2,stride=2),
            nn.ReLU(),

            nn.Conv2d(32,16,kernel_size=5,stride=1,padding=2),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(16,16,kernel_size=2,stride=2),
            nn.ReLU(),

            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),

            nn.Conv2d(16,16,kernel_size=5,stride=1,padding=2),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),

            nn.Conv2d(16,in_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )


    @property
    def loss(self):
        return self.loss_E + 0.001 * self.loss_C

    def forward(self, im_data,gt_map):
        x = self.FME(im_data)
        es_map = self.DME(x)

        if self.training:
            self.loss_E = self.LE(es_map, gt_map)
            self.loss_C = 1 - self.SSIM(es_map, gt_map)
            # print ('loss_E: %10.f , loss_C:%.5f'%(self.loss_E.item(),self.loss_C.item()))
            # self.loss_C = self.ssim_loss(es_map,gt_map)

        return es_map



class SANetModule(nn.Module):
    def __init__(self,in_channels,out_channels,level):
        super(SANetModule,self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.half_channel = int(self.in_channels/2)
        self.level = level
        if self.level !=1:
            self.branch1 = nn.Sequential(
                    nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1,stride=1,padding=0),
                    nn.InstanceNorm2d(self.out_channels,affine=True),
                    nn.ReLU()
            )

            self.branch2 = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.half_channel, kernel_size=1,stride= 1, padding=0),
                    nn.InstanceNorm2d(self.half_channel, affine=True),
                    nn.ReLU(),

                    nn.Conv2d(self.half_channel,self.out_channels,kernel_size=3,stride=1,padding=1),
                    nn.InstanceNorm2d(self.out_channels, affine=True),
                    nn.ReLU()
            )

            self.branch3 = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.half_channel, 1, 1, 0),
                    nn.InstanceNorm2d(self.half_channel, affine=True),
                    nn.ReLU(),

                    nn.Conv2d(self.half_channel, self.out_channels, kernel_size=5, stride=1, padding=2),
                    nn.InstanceNorm2d(self.out_channels, affine=True),
                    nn.ReLU()
            )
            self.branch4 = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.half_channel, 1, 1, 0),
                    nn.InstanceNorm2d(self.half_channel, affine=True),
                    nn.ReLU(),

                    nn.Conv2d(self.half_channel, self.out_channels, kernel_size=7, stride=1, padding=3),
                    nn.InstanceNorm2d(self.out_channels, affine=True),
                    nn.ReLU()
            )

        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(self.out_channels, affine=True),
                nn.ReLU()
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(self.out_channels, affine=True),
                nn.ReLU()
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(self.out_channels, affine=True),
                nn.ReLU()
            )
            self.branch4 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=7, stride=1, padding=3),
                nn.InstanceNorm2d(self.out_channels, affine=True),
                nn.ReLU()
            )





    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x4 = self.branch4(im_data)
        x = torch.cat((x1, x2, x3,x4), 1)

        return x