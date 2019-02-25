import os
import cv2
import csv
import numpy as np
from scipy.io import loadmat
from util import get_density_map


dataset = 'A'


test_path_csv = '/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/test_data/ground_truth/'
test_img_path = '/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/test_data/images/'
out_img_path = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/test_B/images/'
out_den_path = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/test_B/den/'


for i in [test_path_csv, test_img_path, out_img_path, out_den_path]:
    if not os.path.exists(i):
        os.makedirs(i)

if dataset == 'A':
    num_images = 182
else:
    num_images = 316

for i in range(1, num_images+1):
    if i % 10 == 0:
        print('Processing {}/{} files'.format(i, num_images))
    image_info = loadmat(''.join((test_path_csv, 'GT_IMG_', str(i), '.mat')))['image_info']
    input_img_name = ''.join((test_img_path, 'IMG_', str(i), '.jpg'))
    im = cv2.imread(input_img_name, 0)
    annPoints =  image_info[0][0][0][0][0] - 1
    im_density = get_density_map.get_density_map_gaussian(im, annPoints)

    oh,ow = im.shape
    oh = int(oh/8)*8   #这是保证conv之后再deconv所得到的密度图和原图一样大小
    ow = int(ow/8)*8
    c1ws = int(ow * 3 / 8)
    c2ws = int(ow * 5 / 8)
    c1hb = int(oh/2)

    im_sample1 = im[0:c1hb,0:c1ws]
    im_sample2 = im[0:c1hb,c1ws:c2ws]
    im_sample3 = im[0:c1hb,c2ws:ow]
    im_sample4 = im[c1hb:oh,0:c1ws]
    im_sample5 = im[c1hb:oh,c1ws:c2ws]
    im_sample6 = im[c1hb:oh,c2ws:ow]

    den_sample1 = im_density[0:c1hb, 0: c1ws]
    den_sample2 = im_density[0:c1hb, c1ws:c2ws]
    den_sample3 = im_density[0:c1hb, c2ws:ow]
    den_sample4 = im_density[c1hb:oh, 0:c1ws]
    den_sample5 = im_density[c1hb:oh, c1ws:c2ws]
    den_sample6 = im_density[c1hb:oh, c2ws:ow]




    cv2.imwrite(''.join([out_img_path, str(i)+'_1', '.jpg']), im_sample1)
    cv2.imwrite(''.join([out_img_path, str(i) + '_2', '.jpg']), im_sample2)
    cv2.imwrite(''.join([out_img_path, str(i) + '_3', '.jpg']), im_sample3)
    cv2.imwrite(''.join([out_img_path, str(i) + '_4', '.jpg']), im_sample4)
    cv2.imwrite(''.join([out_img_path, str(i) + '_5', '.jpg']), im_sample5)
    cv2.imwrite(''.join([out_img_path, str(i) + '_6', '.jpg']), im_sample6)


    with open(''.join([out_den_path, 'IMG', str(i)+'_1', '.csv']), 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(den_sample1)
    with open(''.join([out_den_path, 'IMG', str(i) + '_2', '.csv']), 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(den_sample2)
    with open(''.join([out_den_path, 'IMG', str(i)+'_3', '.csv']), 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(den_sample3)
    with open(''.join([out_den_path, 'IMG', str(i) + '_4', '.csv']), 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(den_sample4)
    with open(''.join([out_den_path, 'IMG', str(i) + '_5', '.csv']), 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(den_sample5)
    with open(''.join([out_den_path, 'IMG', str(i) + '_6', '.csv']), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(den_sample6)

