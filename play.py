import torch
import torch.nn as nn
from src.SSIM import SSIM
import cv2
import numpy as np
a =cv2.imread('./IMG_3.jpg')
b =cv2.imread('./IMG_6.jpg')
c =cv2.imread('./IMG_113.jpg')

a = a.transpose([2, 0, 1])
a = a[np.newaxis, ...]

b = b.transpose([2, 0, 1])
b = b[np.newaxis, ...]

copy = a
copy += 3

ssim =  SSIM(in_channel=3 , window_size=11,size_average=True)
loss1 = ssim(torch.Tensor(a),torch.Tensor(b))
print (loss1)
