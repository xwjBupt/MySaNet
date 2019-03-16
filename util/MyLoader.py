# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import random
import pandas as pd
import sys


class ImageDataLoader():
    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False,raw = False,num_cut = 2,geo = False):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        # num_classes: total number of classes into which the crowd count is divided (default: 10 as used in the paper)
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = ['GT_' + name.replace('jpg', 'mat') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw
        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx % 30 == 0:
                print('loaded %d imgs' % idx)
            imname = os.path.join(self.imdir, self.imname[idx])

            img = cv2.imread(imname)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img /= 255.0
            img -= np.array(PIXEL_MEANS)
            img /= np.array(PIXEL_STDS)
            gtname = os.path.join(self.gtdir, self.gtname[idx])
            image_info = loadmat(gtname)['image_info']
            annPoints = image_info[0][0][0][0][0] - 1

            if self.test or self.raw:
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                img = cv2.resize(img, (int(w / 16) * 16, int(h / 16) * 16), interpolation=cv2.INTER_LANCZOS4)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img, annPoints)

                img = img.transpose([2, 0, 1])
                den = den.transpose([2, 0, 1])

                self.imgs.append(img)
                self.gts.append(den)

            if self.train and not self.raw:
                self.imgs.append(img)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img, annPoints)
                self.gts.append(den)
    # def get_classifier_weights(self):
    #     # since the dataset is imbalanced, classifier weights are used to ensure balance.
    #     # this function returns weights for each class based on the number of samples available for each class
    #     wts = self.count_class_hist
    #     wts = 1 - wts / (sum(wts))  # 把数据较多的类用较小的权重，所以是1-wts/(sum(wts))
    #     wts = wts / sum(wts)
    #     return wts

    def preload_data(self):
        print('Pre-loading the data. This may take a while...')
        idx = 0
        for fname in self.data_files:
            img, den, gt_count = self.read_image_and_gt(fname)
            self.min_gt_count = min(self.min_gt_count, gt_count)
            self.max_gt_count = max(self.max_gt_count, gt_count)

            blob = {}
            blob['data'] = img
            blob['gt_density'] = den
            blob['fname'] = fname
            blob['gt_count'] = gt_count

            self.blob_list[idx] = blob
            idx = idx + 1
            if idx % 100 == 0:
                print('Loaded ', idx, '/', self.num_samples)
        print('Completed laoding ', idx, 'files')

    # def assign_gt_class_labels(self):  # 给每张图片分密度级别标签
    #     for i in range(0, self.num_samples):
    #         gt_class_label = np.zeros(self.num_classes, dtype=np.int)
    #         bin_val = (self.max_gt_count - self.min_gt_count) / float(self.num_classes)  # 根据分多少类来确定分类间隔
    #         class_idx = np.round(self.blob_list[i]['gt_count'] / bin_val)  # 根据gt_count数目给图片分密度等级
    #         class_idx = int(min(class_idx, self.num_classes - 1))
    #         gt_class_label[class_idx] = 1
    #         self.blob_list[i]['gt_class_label'] = gt_class_label.reshape(1, gt_class_label.shape[0])
    #         self.count_class_hist[class_idx] += 1

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)

        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:

                fname = files[idx]
                img, den, gt_count = self.read_image_and_gt(fname)
                gt_class_label = np.zeros(self.num_classes, dtype=np.int)
                bin_val = (self.max_gt_count - self.min_gt_count) / float(self.num_classes)
                class_idx = np.round(gt_count / bin_val)
                class_idx = int(min(class_idx, self.num_classes - 1))
                gt_class_label[class_idx] = 1

                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname
                blob['gt_count'] = gt_count
                blob['gt_class_label'] = gt_class_label.reshape(1, gt_class_label.shape[0])

            yield blob

    def get_stats_in_dataset(self):

        min_count = 1000000000000
        max_count = 0
        gt_count_array = np.zeros(self.num_samples)
        i = 0
        for fname in self.data_files:
            den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                              header=None).as_matrix()
            den = den.astype(np.float32, copy=False)
            gt_count = np.sum(den)
            min_count = min(min_count, gt_count)
            max_count = max(max_count, gt_count)
            gt_count_array[i] = gt_count
            i += 1

        self.min_gt_count = min_count
        self.max_gt_count = max_count
        bin_val = (self.max_gt_count - self.min_gt_count) / float(self.num_classes)
        class_idx_array = np.round(gt_count_array / bin_val)

        for class_idx in class_idx_array:
            class_idx = int(min(class_idx, self.num_classes - 1))
            self.count_class_hist[class_idx] += 1

    def get_num_samples(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = cv2.imread(os.path.join(self.data_path, fname), 0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht / 4) * 4
        wd_1 = (wd / 4) * 4
        img = cv2.resize(img, (wd_1, ht_1))
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                          header=None).as_matrix()
        den = den.astype(np.float32, copy=False)
        if self.gt_downsample:
            wd_1 = wd_1 / 4
            ht_1 = ht_1 / 4
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))
        else:
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))

        den = den.reshape((1, 1, den.shape[0], den.shape[1]))
        gt_count = np.sum(den)

        return img, den, gt_count


