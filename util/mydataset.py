import torch
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms as T, utils
from PIL import Image
import random
from matplotlib import pyplot as plt
from scipy.io import loadmat
import math
from util.get_density_map import get_fix_gaussian,get_geo_gaussian
import scipy.misc as misc
import h5py
import glob

class Shanghaitech(Dataset):
    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.im = os.listdir(self.imdir)


    def __len__(self):

        return len(self.im)

    def __getitem__(self, idx):
        im_name = os.path.join(self.imdir,self.im[idx])
        gt_name = os.path.join(self.gtdir,self.im[idx].split('.')[0] + '.csv')


        # img = cv2.imread(im_name,0)
        img = misc.imread(im_name)
        img = img.astype(np.float32, copy=False)

        den = pd.read_csv(gt_name, sep=',',
                              header=None).as_matrix()
        den = den.astype(np.float32, copy=False)

        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht / 4) * 4
        wd_1 = (wd / 4) * 4
        img = cv2.resize(img, (int(wd_1), int(ht_1)))
        # wd_1 = wd_1 / 4
        # ht_1 = ht_1 / 4
        den = cv2.resize(den, (int(wd_1), int(ht_1)))
        den = den * ((wd * ht) / (wd_1 * ht_1))



        if self.train:
            pro = random.random()
            if pro>=self.transform:
                    img = cv2.flip(img, 1)
                    den = cv2.flip(den, 1)
            self.den = den.reshape(1, den.shape[0], den.shape[1])
            self.img = img.reshape(1, img.shape[0], img.shape[1])
            return torch.Tensor(self.img),torch.Tensor(self.den)

        if self.test:

            self.img = img.reshape(1, img.shape[0], img.shape[1])
            self.den = den.reshape(1, den.shape[0], den.shape[1])

            return torch.Tensor(self.img), torch.Tensor(self.den)


class SHTech(Dataset):

    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False,raw = False,num_cut = 4,geo = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = ['GT_'+name.replace('jpg', 'mat') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw
        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx %30 == 0:
                print ('loaded %d imgs'%idx)
            imname = os.path.join(self.imdir,self.imname[idx])


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
                img = cv2.resize(img, (int(w / 8) * 8, int(h / 8) * 8), interpolation=cv2.INTER_LANCZOS4)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)

                img = img.transpose([2, 0, 1])
                den = den.transpose([2,0,1])


                self.imgs.append(img)
                self.gts.append(den)

            if self.train and not self.raw:
                self.imgs.append(img)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)
                self.gts.append(den)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):


        img = self.imgs[idx]
        den = self.gts[idx]

        im_den = []
        im_sam = []
        enough = True



        if self.test or self.raw:

            return torch.Tensor(img),torch.Tensor(den)

        if self.train and not self.raw:

            h= img.shape[0]
            w = img.shape[1]
            c = img.shape[2]
            wn2, hn2 = w / 8, h / 8
            wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8  # 1/8 to the original image size

            a_w, b_w = wn2 + 1, w - wn2  # 1/8+1,7/8
            a_h, b_h = hn2 + 1, h - hn2


            while enough:

                # for j in range(0, self.num_cut):
                    r1 = random.random()            #0<r1<1
                    r2 = random.random()
                    x = math.floor((b_w - a_w) * r1 + a_w) #choose the sample center randomly between (1/8 + r1*7/8) image size
                    y = math.floor((b_h - a_h) * r2 + a_h)
                    x1, y1 = int(x - wn2), int(y - hn2)         #sample offset
                    x2, y2 = int(x + wn2 - 1), int(y + hn2 - 1)

                    im_sampled = img[y1-1:y2, x1-1:x2,:]       #sample the image as the size of offset
                    im_density_sampled = den[y1-1:y2, x1-1:x2,:]           #sample the density map in the according area as the size of offset

                    pro = random.random()
                    if pro>=self.transform:
                            im_sampled = cv2.flip(im_sampled, 1)
                            im_density_sampled = cv2.flip(im_density_sampled, 1)


                    if np.sum(im_density_sampled) > 0:

                        im_density_sampled = im_density_sampled.transpose([2,0,1])
                        im_sampled = im_sampled.transpose([2, 0, 1])
                        im_density_sampled = im_density_sampled[np.newaxis,...]
                        im_sampled = im_sampled[np.newaxis,...]
                        im_den.append(torch.Tensor(im_density_sampled))
                        im_sam.append(torch.Tensor(im_sampled))

                    if len(im_den) >= self.num_cut:
                            enough = False



            self.img = torch.cat(im_sam,0)
            self.den = torch.cat(im_den,0)
            return self.img,self.den


class SDNSHTech(Dataset):

    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False,raw = False,num_cut = 2,geo = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = ['GT_'+name.replace('jpg', 'mat') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw
        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx %30 == 0:
                print ('loaded %d imgs'%idx)
            imname = os.path.join(self.imdir,self.imname[idx])


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
                    den = get_geo_gaussian(img,annPoints)

                img = img.transpose([2, 0, 1])
                den = den.transpose([2,0,1])


                self.imgs.append(img)
                self.gts.append(den)

            if self.train and not self.raw:
                self.imgs.append(img)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)
                self.gts.append(den)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):


        img = self.imgs[idx]
        den = self.gts[idx]

        im_den = []
        im_sam = []
        enough = True



        if self.test or self.raw:

            return torch.Tensor(img),torch.Tensor(den)

        if self.train and not self.raw:

            h= img.shape[0]
            w = img.shape[1]
            c = img.shape[2]
            # wn2, hn2 = w / 4, h / 4
            # wn2, hn2 = int(wn2 / 4) * 4, int(hn2 / 4) * 4  # 1/8 to the original image size
            wn2, hn2 = w / 4, h / 4
            wn2, hn2 = int(wn2 / 16) * 16, int(hn2 / 16) * 16  # 1/8 to the original image size


            a_w, b_w = wn2 + 1, w - wn2  # 1/8+1,7/8
            a_h, b_h = hn2 + 1, h - hn2


            while enough:

                # for j in range(0, self.num_cut):
                    r1 = random.random()            #0<r1<1
                    r2 = random.random()
                    x = math.floor((b_w - a_w) * r1 + a_w) #choose the sample center randomly between (1/8 + r1*7/8) image size
                    y = math.floor((b_h - a_h) * r2 + a_h)
                    x1, y1 = int(x - wn2), int(y - hn2)         #sample offset
                    x2, y2 = int(x + wn2 - 1), int(y + hn2 - 1)

                    im_sampled = img[y1-1:y2, x1-1:x2,:]       #sample the image as the size of offset
                    im_density_sampled = den[y1-1:y2, x1-1:x2,:]           #sample the density map in the according area as the size of offset

                    pro = random.random()
                    if pro>=self.transform:
                            im_sampled = cv2.flip(im_sampled, 1)
                            im_density_sampled = cv2.flip(im_density_sampled, 1)


                    if np.sum(im_density_sampled) > 0:

                        im_density_sampled = im_density_sampled.transpose([2,0,1])
                        im_sampled = im_sampled.transpose([2, 0, 1])
                        im_density_sampled = im_density_sampled[np.newaxis,...]
                        im_sampled = im_sampled[np.newaxis,...]
                        im_den.append(torch.Tensor(im_density_sampled))
                        im_sam.append(torch.Tensor(im_sampled))

                    if len(im_den) >= self.num_cut:
                            enough = False



            self.img = torch.cat(im_sam,0)
            self.den = torch.cat(im_den,0)
            return self.img,self.den



class veichle(Dataset):

    def __init__(self,imdir,gtdir,phase,preload=False,resize = 720,down = 8,raw = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.resize = resize
        self.phase = phase
        self.preload = preload
        self.down_sample = down
        self.raw = raw

        if resize is not None:
            assert self.resize%8==0
        self.imnames = glob.glob(os.path.join(self.imdir,'*.jpg'))
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imnames)
        print('Loading data,wait a second')
        self.PIXEL_MEANS = (0.485, 0.456, 0.406)
        self.PIXEL_STDS = (0.229, 0.224, 0.225)
        self.imgs = []
        self.gts=[]
        self.name = []

        if self.preload:
            for idx in range(self.num_it):
                if idx %100 == 0:
                    print ('loaded %d/%d imgs'%(idx, self.num_it))
                imname = self.imnames[idx]
                self.name.append(imname.split('/')[-1])
                gtname = imname.replace('images','ground_truth').replace('jpg','h5')

                img = cv2.imread(imname)
                img = img[:, :, ::-1].copy()
                img = img.astype(np.float32, copy=False)
                img /= 255.0
                img -= np.array(self.PIXEL_MEANS)
                img /= np.array(self.PIXEL_STDS)
                scale = img.shape[0] / img.shape[1]
                gt = h5py.File(gtname,'r')['density'].value

                if self.resize is not None and self.down_sample > 1:
                    new_height = int(self.resize * scale / 8) * 8
                    rw, rh, _ = img.shape
                    img = cv2.resize(img, (self.resize, new_height))
                    ds_rows = int(img.shape[0] // self.down_sample)
                    ds_cols = int(img.shape[1] // self.down_sample)
                    # zoom = rw*rh / ds_rows / ds_rows
                    den = cv2.resize(gt, (ds_cols, ds_rows),interpolation=cv2.INTER_CUBIC)
                    # den = den * zoom
                    zoom = np.sum(gt) / np.sum(den)
                    den = den * zoom

                if self.resize is not None and self.phase =='test':
                    new_height = int(self.resize * scale / 8) * 8
                    rw, rh, _ = img.shape
                    img = cv2.resize(img, (self.resize, new_height), interpolation=cv2.INTER_CUBIC)
                    den = gt
                    # ds_rows = int(img.shape[0] // self.down_sample)
                    # ds_cols = int(img.shape[1] // self.down_sample)
                    # zoom = img.shape[0] * img.shape[1] / ds_rows / ds_rows
                    # den = cv2.resize(gt, (ds_cols, ds_rows),interpolation=cv2.INTER_CUBIC)
                    # den = den * zoom

                self.imgs.append(img)
                self.gts.append(den)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):

        if self.preload:
            img = self.imgs[idx]
            den = self.gts[idx]
            name = self.name[idx]

        else:
            name = self.imnames[idx]
            img = cv2.imread(self.imnames[idx])
            gtname = self.imnames[idx].replace('jpg', 'h5')

            img = img[:, :, ::-1].copy()
            img = img.astype(np.float32, copy=False)
            img /= 255.0
            img -= np.array(self.PIXEL_MEANS)
            img /= np.array(self.PIXEL_STDS)
            scale = img.shape[0] / img.shape[1]

            gt = h5py.File(gtname, 'r')['density'].value
            rawcount = np.sum(gt)

            if self.resize is not None and self.down_sample and self.phase!='test' >1:
                new_height = int(self.resize * scale / 8) * 8
                rw,rh,_ = img.shape
                img = cv2.resize(img, (self.resize, new_height),interpolation=cv2.INTER_CUBIC)
                # den = cv2.resize(gt, (self.resize, new_height), interpolation=cv2.INTER_CUBIC) *rw*rh/self.resize/new_height

                ds_rows = int(img.shape[0] // self.down_sample)
                ds_cols = int(img.shape[1] // self.down_sample)
                # zoom = img.shape[0]*img.shape[1]/ ds_rows / ds_rows
                # zoom = np.sum(gt) /np.sum(den)
                den = cv2.resize(gt, (ds_cols, ds_rows),interpolation=cv2.INTER_CUBIC)
                zoom = np.sum(gt) / np.sum(den)
                den = den * zoom

            if self.resize is not None and self.phase == 'test':
                new_height = int(self.resize * scale / 8) * 8
                rw, rh, _ = img.shape
                img = cv2.resize(img, (self.resize,new_height), interpolation=cv2.INTER_CUBIC)
                den = gt

                # ds_rows = int(img.shape[0] // self.down_sample)
                # ds_cols = int(img.shape[1] // self.down_sample)
                # zoom = img.shape[0] * img.shape[1] / ds_rows / ds_rows
                # den = cv2.resize(gt, (ds_cols, ds_rows),interpolation=cv2.INTER_CUBIC)
                # den = den * zoom

        if self.phase == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                den = cv2.flip(den, 0)
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    den = cv2.flip(den, 1)

        img = img.transpose([2, 0, 1])
        img = torch.tensor(img)
        den = torch.tensor(den).unsqueeze(0)
        if self.raw:
            return img,den,name
        else:
            return img,den



class veichleDepth(Dataset):

    def __init__(self,imdir,gtdir,phase,preload=False,down = 8,raw = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.phase = phase
        self.preload = preload
        self.down_sample = down
        self.raw = raw


        self.imnames = glob.glob(os.path.join(self.imdir,'*.jpg'))
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imnames)
        print('Loading data,wait a second')
        self.PIXEL_MEANS = (0.485, 0.456, 0.406)
        self.PIXEL_STDS = (0.229, 0.224, 0.225)
        self.imgs = []
        self.gts=[]
        self.name = []

        if self.preload:
            for idx in range(self.num_it):
                if idx %100 == 0:
                    print ('loaded %d/%d imgs'%(idx, self.num_it))

                imname = self.imnames[idx]
                gtname = self.gtdir + self.imnames[idx].split('/')[-1].replace('.jpg', '_mega_gt.h5')
                img = cv2.imread(imname)
                img = img[:, :, ::-1].copy()
                img = img.astype(np.float32, copy=False)
                img /= 255.0
                img -= np.array(self.PIXEL_MEANS)
                img /= np.array(self.PIXEL_STDS)
                rh,rw,_ = img.shape
                img = cv2.resize(img,(1024,768),interpolation=cv2.INTER_CUBIC)


                gt = h5py.File(gtname,'r')['density'].value

                if self.down_sample > 1:
                    ds_rows = int(gt.shape[0] // self.down_sample)
                    ds_cols = int(gt.shape[1] // self.down_sample)
                    den = cv2.resize(gt, (ds_cols, ds_rows), interpolation=cv2.INTER_CUBIC)
                    zoom = np.sum(gt) / np.sum(den)
                    den = den * zoom

                if self.phase =='test':
                    rw, rh, _ = img.shape
                    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_CUBIC)
                    den = gt


                self.imgs.append(img)
                self.gts.append(den)
                self.name.append(imname)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):

        if self.preload:
            img = self.imgs[idx]
            den = self.gts[idx]
            name = self.name[idx]

        else:
            name = self.imnames[idx]
            img = cv2.imread(self.imnames[idx])
            # gtname = self.gtdir+self.imnames[idx].split('/')[-1].replace('.jpg','_mega_gt.h5')
            gtname = self.gtdir + self.imnames[idx].split('/')[-1].replace('.jpg', '_mega_gt.h5')
            img = img[:, :, ::-1].copy()
            img = img.astype(np.float32, copy=False)
            img /= 255.0
            img -= np.array(self.PIXEL_MEANS)
            img /= np.array(self.PIXEL_STDS)
            img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_CUBIC)



            gt = h5py.File(gtname, 'r')['density'].value
            rawcount = np.sum(gt)

            if self.down_sample>1 and self.phase!='test' :

                ds_rows = int(gt.shape[0] // self.down_sample)
                ds_cols = int(gt.shape[1] // self.down_sample)

                # zoom = gt.shape[0]*gt.shape[1]/ ds_rows / ds_rows

                den = cv2.resize(gt, (ds_cols, ds_rows),interpolation=cv2.INTER_CUBIC)
                zoom = rawcount/np.sum(den)
                den = den * zoom

            if  self.phase == 'test':
                img = cv2.resize(img, (1024,768), interpolation=cv2.INTER_CUBIC)
                den = gt

        if self.phase == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                den = cv2.flip(den, 0)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                den = cv2.flip(den, 1)

        img = img.transpose([2, 0, 1])
        img = torch.tensor(img)
        den = torch.tensor(den).unsqueeze(0)


        return img,den,name


if __name__ == '__main__':
    trainim_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/original/train_set/images/'
    traingt_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/new_mega_gt/train/'
    train_data = veichleDepth(trainim_file, traingt_file, preload=False,phase='train',raw=True)
    train_loader = DataLoader(train_data,batch_size=1,shuffle=False,num_workers=0)
    diff = 0.
    plt.ion()  # 开启interactive mode

    for i ,(img,gt,raw,name) in enumerate(train_loader):
        img = img[0][0]
        gt = gt[0][0]
        gtcount = gt.sum().item()
        diff +=abs(raw.item()-gtcount)
        print ('name: %s raw: %.3f @ %.3f :gtcount - diff: %.3f'%(name,raw.item(),gtcount,raw.item()-gtcount))

        plt.subplot(121)
        plt.imshow(img)
        plt.title(raw.item())

        plt.subplot(122)
        plt.imshow(gt)
        plt.title(gtcount)

        plt.suptitle(name[0].split('/')[-1])
        # plt.show()
        plt.pause(0.5)  # 显示秒数

        plt.close()
