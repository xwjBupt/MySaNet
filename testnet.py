import torch
import torch.nn as nn
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


logger = logging.getLogger(name='train')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('output.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
        '[%(asctime)s]-[module:%(name)s]-[line: %(lineno)d]:%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)




if __name__ == '__main__':


    lr = 0.0001
    epochs = 25
    step = 0

    escount = []
    gtcount = []

    bestloss = float('inf')
    bestmae = float('inf')
    dataset = 'ShTA-cut'
    method = 'SANet'
    resume = True
    startepoch = 0
    writer = SummaryWriter()
    trainim_file = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/train_A/images'
    traingt_file = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/train_A/den'
    valim_file = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/val_A/images'
    valgt_file = '/media/xwj/Data/DataSet/shanghai_tech/sanet/part_A_final/val_A/den'

    train_data = Shanghaitech(imdir = trainim_file,gtdir=traingt_file,transform= 0.5,train=True,test = False)
    val_data = Shanghaitech(imdir = valim_file,gtdir=valgt_file,train = False,test = True)
    train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=8)



    net = SANet()
    cprint('=> loading checkpoint : ./saved_models/best_loss.tar',color='yellow')
    checkpoint = torch.load('./saved_models/best_loss.tar')

    best_loss = checkpoint['best_loss']
    net.load_state_dict(checkpoint['state_dict'])

    cprint("=> loaded checkpoint ",color='yellow')


    net.cuda()
    net.train()

    valloss = AverageMeter()
    valstart = time.time()
    step = 0.
    valmse = 0.
    valmae = 0.
    for index,(timg,tden) in tqdm(enumerate(val_loader)):
            step +=1
            timg = timg.cuda()
            tden = tden.cuda()
            tes_den = net(timg, tden)
            tloss = net.loss
            valloss.update(tloss.item(),timg.shape[0])
            tes_count = np.sum(tes_den[0][0].cpu().detach().numpy())
            tgt_count = np.sum(tden[0][0].cpu().detach().numpy())

            escount.append(tes_count)
            gtcount.append(tgt_count)

            mse = abs(tes_count - tgt_count)
            mae = 0.5*mse * mse

            writer.add_scalars('data/valstate', {
                'mse': mse,
                'mae': mae}, step)

            writer.add_scalar('valloss', valloss.avg, step)
            cprint ('valloss:{} - mse:{} - mse:{}'.format(valloss.avg,mse,mae))


            if step % 50==0 :
                plt.subplot(131)
                plt.title('raw image')
                plt.imshow(timg[0][0].cpu().detach().numpy())

                plt.subplot(132)
                gt = np.sum(tden[0][0].cpu().detach().numpy())
                plt.title('gtcount:%.2f'%gt)
                plt.imshow(tden[0][0].cpu().detach().numpy())


                plt.subplot(133)
                es = np.sum(tes_den[0][0].cpu().detach().numpy())
                plt.title('escount:%.2f'%es)
                plt.imshow(tes_den[0][0].cpu().detach().numpy())
                plt.show()

    durantion = time.time()-valstart
    valfps = step/durantion
    for i in range(len(escount)):
        temp1 = abs(escount[i] - gtcount[i])
        temp2 = temp1 * temp1
        valmae += temp1
        valmse += temp2
    valmae *= 1. / len(val_data)
    valmse = math.sqrt(1. / len(val_data) * valmse)
    cprint('valfps:{} - valmse:{} - valmse:{}'.format(valfps,valmse, valmae))
    cprint('Done',color='green')












