import torch
import torch.nn as nn
import torchvision.utils as vutil
from util.lib import *
import sys
from util.mydataset import *
from src.Net import *
from src.net import SANet as newSa
import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import math
from src.ssim_loss import *
from tqdm import tqdm
from util.mcnn_loader import ImageDataLoader
from src.MCNN import *
from util.Timer import Timer
from util.lr import *
from util.lib import eva_model
import shutil
from util.mydataset import *
from src.can import *
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from src.copyloss import *


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


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    current_dir = os.getcwd()
    method = 'whole_val_test'

    saveimg = current_dir + '/' + method + '/img/'
    savelog = current_dir + '/' + method + '/'
    need_dir = [saveimg,  savelog]
    for i in need_dir:
        if not os.path.exists(i):
            os.makedirs(i)

    logger = logging.getLogger(name='test')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(savelog + 'output_test_depth_gt.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s]-[module:%(name)s]-[line: %(lineno)d]:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    testim_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/temp/val-test/'
    testgt_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/temp/val-test/'

    test_data = veichleDepth(testim_file, testgt_file, preload=False,  phase='test', raw=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    net = CANNet()
    cprint('=> loading checkpoint : whole_val_test/best_loss_mae_can_zero_depth_gt.pth ', color='yellow')
    checkpoint = torch.load('whole_val_test/best_loss_mae_can_zero_depth_gt.pth')
    startepoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    net.load_state_dict(checkpoint['state_dict'])

    cprint("=> loaded checkpoint ", color='yellow')
    net.cuda()
    net.eval()
    escount = []
    gtcount = []
    logger.info('\n')
    logger.info('@@@@@@ START TEST AT : %s @@@@@' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    duration = 0.0
    recod = {}

    with torch.no_grad():
        for index, (img, den, name) in tqdm(enumerate(test_loader)):
            img = img.cuda()
            den = den.cuda()
            start = time.time()
            es_den = net(img)
            name =name[0].split('/')[-1]
            es_count = np.sum(es_den[0][0].cpu().detach().numpy())
            gt_count = np.sum(den[0][0].cpu().detach().numpy())

            diff = abs(es_count - gt_count)
            stop = time.time()

            logger.info(name+' @ gt: %.3f vs %.3f:es @diff: %.3f' % (gt_count, es_count, diff))
            duration += stop - start
            recod[name] = diff
            escount.append(es_count)
            gtcount.append(gt_count)

            # plt.subplot(131)
            # plt.title(name)
            # plt.imshow(img[0][0].cpu().detach().numpy())
            # plt.subplot(132)
            # plt.title('gtcount:%.2f' % gt_count)
            # plt.imshow(den[0][0].cpu().detach().numpy())
            # plt.subplot(133)
            # plt.title('escount:%.2f' % es_count)
            # plt.imshow(es_den[0][0].cpu().detach().numpy())
            # plt.savefig(saveimg + name)
            # cprint (saveimg+name,color='yellow')

    list1 = sorted(recod.items(), key=lambda x: x[1],reverse = True)

    fps = len(test_data)/duration
    MAE, MSE = eva_model(escount, gtcount)
    logger.info('$$$$ MAE: %.3f - MSE: %.3f - FPS: %.3f $$$$' % (MAE, MSE, fps))

    logger.info('\n')
    logger.info('!!!! sort start !!!!')
    for i in list1:
        logger.info(i[0]+' @ '+str(i[1]))
    logger.info('!!!! sort done !!!!')