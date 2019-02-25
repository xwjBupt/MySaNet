import torch
import torch.nn as nn
from src.network_mcnn import Conv2d as Conv2d
import src.network_mcnn as network
from matplotlib import pyplot as plt
import time

cout = 0

class MCNN(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        self.count = 0
        self.branch1_1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))

        self.branch1_2 = nn.Sequential(
            Conv2d(16, 32, 7, same_padding=True, bn=bn),
            nn.MaxPool2d(2)
        )

        self.branch1_3 = nn.Sequential(
            Conv2d(32, 16, 7, same_padding=True, bn=bn),
            Conv2d(16, 8, 7, same_padding=True, bn=bn)
        )



        self.branch2_1 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))

        self.branch2_2 = nn.Sequential(
            Conv2d(20, 40, 5, same_padding=True, bn=bn),
            nn.MaxPool2d(2)
        )

        self.branch2_3 = nn.Sequential(
            Conv2d(40, 20, 5, same_padding=True, bn=bn),
            Conv2d(20, 10, 5, same_padding=True, bn=bn)
        )



        self.branch3_1 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.branch3_2 =nn.Sequential(
            Conv2d(24, 48, 3, same_padding=True, bn=bn),
            nn.MaxPool2d(2)
        )

        self.branch3_3 = nn.Sequential(
            Conv2d(48, 24, 3, same_padding=True, bn=bn),
            Conv2d(24, 12, 3, same_padding=True, bn=bn)
        )

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data,gt_data):
        x1_1 = self.branch1_1(im_data)
        x1_2 = self.branch1_2(x1_1)
        x1 = self.branch1_3(x1_2)

        x2_1 = self.branch2_1(im_data)
        x2_2 = self.branch2_2(x2_1)
        x2 = self.branch2_3(x2_2)

        x3_1 = self.branch3_1(im_data)
        x3_2 = self.branch3_2(x3_1)
        x3 = self.branch3_3(x3_2)

        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        # if self.count %500 == 0:
        #     # plt.ion()
        #     plt.subplot(321)
        #     plt.imshow(x2_1[0][0].cpu().detach().numpy())
        #     plt.subplot(322)
        #     plt.imshow(x2_2[0][0].cpu().detach().numpy())
        #     plt.subplot(323)
        #     plt.imshow(x2[0][0].cpu().detach().numpy())
        #     plt.subplot(324)
        #     plt.imshow(im_data[0][0].cpu().detach().numpy())
        #     plt.subplot(325)
        #     plt.imshow(x[0][0].cpu().detach().numpy())
        #     plt.subplot(326)
        #     plt.imshow(gt_data[0][0].cpu().detach().numpy())
        #
        #     # plt.pause(3)
        #     # plt.ioff()
        self.count +=1

        return x




class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.DME(im_data,gt_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss
