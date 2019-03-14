import torch
import torch.nn as nn
import torchvision as vision
class TEST(nn.Module):

    def __init__(self):
        super(TEST,self).__init__()
        dense = vision.models.densenet121(pretrained=False)
        print (dense)
        backbone1 = nn.Sequential(dense.features[0:4])
        backbone2 = nn.Sequential(dense.features[4:6])
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module('Backbone1', backbone1)
        self.encoder.add_module('Backbone2', backbone2)
        self.decoder.add_module('finalconv1',
                                nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1))
        self.decoder.add_module('finalconv2', nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1))

    def forward(self, *input):
        pass

a = TEST()
print (a)

# import matplotlib.pyplot as plt
# import os
# imdir = '/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/train_data/images'
# imnames = os.listdir(imdir)
# import cv2
# xh = []
# xw = []
#
# for i in range(len(imnames)):
#     imname = os.path.join(imdir,imnames[i])
#     img = cv2.imread(imname)
#     h,w,c = img.shape
#
#     # nh = int (h/4)*4
#     # nw = int(w/4)*4
#     #
#     fh = h/64
#     fw = w / 64
#     xh.append(fh)
#     xw.append(fw)
#
#
# plt.scatter(xh,xw,s=200)
#
# plt.title('X',fontsize=24)
# plt.xlabel("XH",fontsize=14)
# plt.xlabel("XW",fontsize=14)
#
# #设置刻度标记的大小
# plt.tick_params(axis='both',which='major',labelsize=14)
#
# #设置每个坐标的取值范围
# plt.axis([0,50,0,50])
#
# plt.show()