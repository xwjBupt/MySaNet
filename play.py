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

