import glob
import os
from termcolor import cprint
import torch
import torch.nn as nn
import torchvision.utils as vutil
import sys
import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import math

from tqdm import tqdm
import shutil
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import logging
import torch.utils.data.dataloader as dataloader
from cannet import *
from my_dataset import *
import time
from lib import *
import argparse
from ucfcc50 import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.LE = nn.MSELoss(reduction='elementwise_mean')
        # self.SSIM = SSIM(in_channel=1, window_size=11, size_average=True)

    def forward(self, es_map, gt_map):
        self.loss_E = self.LE(es_map, gt_map)
        # self.loss_C = 1 - self.SSIM(es_map, gt_map)
        # my_loss = 0.001 * self.loss_E + self.loss_C

        return self.loss_E


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def train(net,train_loader,optimizer,dis,trainloss,Loss,**kwargs):
    trainstart = time.time()
    step = 0.
    net.train()
    escounts = []
    gtcounts = []
    length = len(train_loader)

    for index, (img, den, name) in tqdm(enumerate(train_loader)):
        step += 1
        img = img.cuda()
        den = den.cuda()
        es_den = net(img)

        loss = Loss(den, es_den)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss.update(loss.item(), img.shape[0])

        es_count = np.sum(es_den[0][0].cpu().detach().numpy())
        gt_count = np.sum(den[0][0].cpu().detach().numpy())

        if index%dis ==0:
            diff = abs(gt_count-es_count)
            cprint ('[%d\t/%d][trainloss*10e3: %.5f][es: %.4f - gt: %.4f @diff:%.4f]'%(index,length,trainloss.avg*1000,es_count,gt_count,diff),color='yellow')

        escounts.append(es_count)
        gtcounts.append(gt_count)

    durantion = time.time() - trainstart
    trainfps = step / durantion

    trainmae, trainmse = eva_model(escounts, gtcounts)
    return net,trainmae,trainmse,trainfps,trainloss

def val(net,val_loader,valloss,epoch,saveimg,valLoss,**kwargs):
    with torch.no_grad():
        net.eval()
        time_stamp = 0.0
        escounts = []
        gtcounts = []
        plt.ion()
        for index, (timg, tden, tname) in tqdm(enumerate(val_loader)):
            start = time.time()
            timg = timg.cuda()
            tden = tden.cuda()
            tes_den = net(timg)
            tname = tname[0].split('.')[0]
            tloss = valLoss(tes_den, tden)

            valloss.update(tloss.item(), timg.shape[0])

            tes_count = np.sum(tes_den[0][0].cpu().detach().numpy())
            tgt_count = np.sum(tden[0][0].cpu().detach().numpy())

            escounts.append(tes_count)
            gtcounts.append(tgt_count)

            durantion = time.time() - start
            time_stamp += durantion


            if epoch % args.epoch_dis == 0:

                plt.subplot(131)
                plt.title('image')
                plt.imshow(timg[0][0].cpu().detach().numpy())

                plt.subplot(132)
                plt.title('gtcount:%.2f' % tgt_count)
                plt.imshow(tden[0][0].cpu().detach().numpy())

                plt.subplot(133)
                plt.title('escount:%.2f' % tes_count)
                plt.imshow(tes_den[0][0].cpu().detach().numpy())

                if index % args.iter_dis == 0:
                    plt.savefig(saveimg + '/%s-epoch%d.jpg' % (tname, epoch))
                    plt.pause(0.5)

        plt.close()
        plt.ioff()

    valfps = len(val_loader) / time_stamp

    valmae, valmse = eva_model(escounts, gtcounts)

    return valmae,valmse,valfps,valloss


if __name__ == '__main__':

    parser = argparse.ArgumentParser('setup record')
    parser.add_argument("--method", default='CAN',help='raw can')
    parser.add_argument("--dataset", default='UCF50/set2')
    parser.add_argument("--bs", default=1)
    parser.add_argument("--lr", default=1e-7)
    parser.add_argument("--epochs", default=300)
    parser.add_argument("--finetune", default=False)
    parser.add_argument("--resume", default=True)
    parser.add_argument("--epoch_dis", default=30)
    parser.add_argument("--iter_dis", default=5)
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument("--momentum", default=0.95)
    parser.add_argument("--best_mae", default=float('inf'))
    parser.add_argument("--best_loss", default=float('inf'))
    parser.add_argument("--best_mae_epoch", default=-1)
    parser.add_argument("--best_loss_epoch", default=-1)
    parser.add_argument("--best_mse", default=float('inf'))
    parser.add_argument("--works", default=4)
    parser.add_argument("--show_model", default=True)
    parser.add_argument("--test", default=False)
    parser.add_argument("--lr_changer",default=None,choices=['cosine','step','expotential','rop','cosann',None])
    args = parser.parse_args()

    if args.bs >1:
        torch.backends.cudnn.benchmark = True


    current_dir = os.getcwd()
    saveimg = current_dir + '/' + args.dataset + '/' + args.method + '/img/'
    savemodel = current_dir + '/' + args.dataset + '/' + args.method + '/model/'
    savelog = current_dir + '/' + args.dataset + '/' + args.method + '/'
    ten_log = current_dir +'/'+  args.dataset + '/' + args.method + '/runs/'

    need_dir = [saveimg, savemodel, savelog,ten_log]
    for i in need_dir:
        if not os.path.exists(i):
            os.makedirs(i)

    writer = SummaryWriter(log_dir=ten_log,comment='set2')


    logger = logging.getLogger(name='train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(savelog + 'output.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s]:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('\n\n\n')
    logger.info('@@@@@@ START  RUNNING  AT : %s @@@@@' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(args)
    logger.info('\n')

    val_folder = 2
    train_folder = get_train_folder(val_folder)

    # trainim_file = '/media/xwj/Data/DataSet/ShanghaiTech/original/part_A/train_data/images/'
    # traingt_file = '/media/xwj/Data/DataSet/ShanghaiTech/SHTA-Depth/GT-Depth/part_A/train_data/'
    # valim_file = '/media/xwj/Data/DataSet/ShanghaiTech/original/part_A/test_data/images/'
    # valgt_file = '/media/xwj/Data/DataSet/ShanghaiTech/SHTA-Depth/GT-Depth/part_A/test_data/'
    data_root = '/media/xwj/Data/DataSet/UCFCC50/processed'

    train_data = MYUCF50(data_root,train_folder,phase='train')
    val_data = MYUCF50(data_root, map(int, str(val_folder)),phase='test')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.works)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=args.works)

    net = CANNet()
    net.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0.9)


    if args.show_model:
        torch.save(net,'model.pth')


    if args.resume:
        if args.test or args.finetune:
            model =savemodel + 'best_mae.pth'
        else:
            model = savemodel + 'last_check.pth'

        cprint('=> loading checkpoint : %s ' % model, color='yellow')
        checkpoint = torch.load(model)
        args.start_epoch = checkpoint['epoch']
        args.best_loss = checkpoint['best_loss']
        args.best_mae = checkpoint['best_mae']
        args.lr = checkpoint['lr']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.load_state_dict(checkpoint['net_state_dict'])
        cprint("=> loaded checkpoint ", color='yellow')
        state = 'epoch:%d lr:%.8f best_mae:%.4f best_loss:%.10f\n'%(args.start_epoch,args.lr,args.best_mae, args.best_loss)
        logger.info('resume state: '+state)


    if args.lr_changer =='step':
        scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
    elif args.lr_changer =='cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    elif args.lr_changer =='expotential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
    elif args.lr_changer =='rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    elif args.lr_changer=='cosann':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10)
    else:
        scheduler = None

    Loss = Myloss()
    valLoss = Myloss()


    for epoch in range(args.start_epoch, args.epochs):
        trainloss = AverageMeter()
        valloss = AverageMeter()

        LR = get_learning_rate(optimizer)
        logger.info('epoch:{} -- lr*10000:{}'.format(epoch, LR * 10000))

        ##train
        cprint('start train', color='yellow')
        net,trainmae,trainmse,trainfps,trainloss = train(net,train_loader,optimizer,args.iter_dis,trainloss,Loss)

        info = 'trianloss*10e3:%.5f @ trainmae:%.4f @ trainmse:%.4f @ trainfps:%.3f' % (
            trainloss.avg * 1000, trainmae, trainmse, trainfps)
        logger.info(info)

        writer.add_scalars('trainstate', {
            'trainloss': trainloss.avg,
            'trainmse': trainmse,
            'trainmae': trainmae
        }, epoch)


        #val
        cprint('start val',color='yellow')
        valmae,valmse,valfps,valloss = val(net,val_loader,valloss,epoch,saveimg,valLoss)

        info = 'valloss*10e3:%.5f @ valmae:%.4f @ valmse:%.4f @ valfps%.3f' % (
            valloss.avg * 1000,
            valmae,
            valmse,
            valfps
        )
        logger.info(info)

        writer.add_scalars('valstate', {
            'valloss': valloss.avg,
            'valmse': valmse,
            'valmae': valmae,
        }, epoch)

        if args.lr_changer =='rop':
            scheduler.step(valmae)
        elif args.lr_changer is not None:
            scheduler.step()
        else:
            pass

        save_dict = {
            'epoch': epoch + 1,
            'net_state_dict': net.state_dict(),
            'best_loss': args.best_loss,
            'best_mae': args.best_mae,
            'best_loss_epoch':args.best_loss_epoch,
            'best_mae_epoch':args.best_mae_epoch,
            'lr': get_learning_rate(optimizer),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(save_dict, savemodel + 'last_check.pth')

        if args.best_mae > valmae:
            args.best_mae = valmae
            args.best_mae_epoch = epoch
            shutil.copy(savemodel + 'last_check.pth', savemodel + 'best_mae.pth')

        if args.best_loss > valloss.avg:
            args.best_loss = valloss.avg
            args.best_loss_epoch = epoch
            shutil.copy(savemodel + 'last_check.pth', savemodel + 'best_loss.pth')

        crruent = '[best mae: %.4f @ epoch: %d] - [best loss*10e3: %.5f @ epoch: %d] \n'%(args.best_mae,args.best_mae_epoch,args.best_loss*1000,args.best_loss_epoch)
        logger.info(crruent)


    logger.info(args.method + ' train complete')
    crruent = '[best mae: %.4f @ epoch: %d] - [best loss*10e3: %.5f @ epoch: %d]' % (
    args.best_mae, args.best_mae_epoch, args.best_loss * 1000, args.best_loss_epoch)
    logger.info(crruent)
    logger.info('save bestmodel to ' + savemodel)
