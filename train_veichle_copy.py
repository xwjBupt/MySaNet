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
        self.SSIM = SSIM(in_channel=1, window_size=11, size_average=True)

    def forward(self,es_map,gt_map):
        self.loss_E = self.LE(es_map, gt_map)
        self.loss_C = 1 - self.SSIM(es_map, gt_map)
        my_loss = 0.001*self.loss_E + self.loss_C

        return my_loss



def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    save = 50
    log_interval = 250
    lr = 0.000001
    epochs = 30
    step = 0
    valloss = 0.
    escount = []
    gtcount = []
    vallossdata = 0.
    trainlossdata = 0.
    best_loss = float('inf')
    best_mae = float('inf')
    best_mse = float('inf')
    finetune = False
    dataset = 'vechile'
    method = 'can_1080_zero_newgt'
    resume = False
    startepoch = 0
    current_dir = os.getcwd()
    saveimg = current_dir+'/'+method+'/img'
    savemodel = current_dir+'/'+method+'/model'
    savelog = current_dir+'/'+method+'/'
    ten_log = current_dir+'/runs/'+method
    # if os.path.exists(ten_log):
    #     shutil.rmtree(ten_log)
    need_dir = [saveimg,savemodel,savelog]
    for i in need_dir:
        if not os.path.exists(i):
            os.makedirs(i)

    writer = SummaryWriter(log_dir=ten_log)

    logger = logging.getLogger(name='train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(savelog+'output.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s]-[module:%(name)s]-[line: %(lineno)d]:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    trainim_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/train_set/images'
    traingt_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/train_set/ground_truth'
    valim_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/val_set/images'
    valgt_file = '/media/xwj/xwjdata1TA/Dataset/vehicle/val_set/ground_truth'



    train_data = veichle(trainim_file,traingt_file,preload=False,resize=1920,phase = 'train')
    val_data = veichle(valim_file,valgt_file,preload=False,resize=1920,phase = 'val')

    train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=0)
    val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=0)


    net = CANNet()


    if resume:
        cprint('=> loading checkpoint : ./rgb-randomcrop/model/modelbest_loss_cut_rgb-randomcrop.tar ',color='yellow')
        checkpoint = torch.load(current_dir +'/rgb-randomcrop/model/modelbest_loss_cut_rgb-randomcrop.tar')
        startepoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['state_dict'])
        # lr = checkpoint['lr']
        cprint("=> loaded checkpoint ",color='yellow')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    net.cuda()
    net.train()

    LOSS = Myloss()
    VALLOSS = Myloss()
    logger.info('\n\n\n')
    logger.info('@@@@@@ START  TRAIN  AT : %s @@@@@'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    logger.info(method)
    for epoch in range(0,epochs):
        trainmae = 0.
        trainmse = 0.
        valmae =0.
        valmse =0.
        trainloss = AverageMeter()
        valloss = AverageMeter()
        # LR = adjust_learning_rate(optimizer, epochs, epoch, lr)
        LR = get_learning_rate(optimizer)
        logger.info('epoch:{} -- lr*10000:{}'.format(epoch,LR*10000))
        scheduler.step()
        trainstart = time.time()
        step = 0.
        net.train()
        for index,(img,den) in tqdm(enumerate(train_loader)):
            step +=1
            img = img.cuda()
            den = den.cuda()

            es_den = net(img)

            loss = LOSS(den,es_den)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainloss.update(loss.item(), img.shape[0])


            es_count = np.sum(es_den[0][0].cpu().detach().numpy())
            gt_count = np.sum(den[0][0].cpu().detach().numpy())
            escount.append(es_count)
            gtcount.append(gt_count)
        durantion = time.time()-trainstart
        trainfps = step/durantion

        trainmae,trainmse = eva_model(escount,gtcount)
        writer.add_scalars('data/trainstate', {
                                          'trainmse': trainmse,
                                          'trainmae': trainmae}, epoch)

        info = 'trianloss:{%.6f} @ trainmae:{%.3f} @ trainmse:{%.3f} @ fps:{%.3f}'%(trainloss.avg*10000,trainmae, trainmse,trainfps)
        logger.info(info)

        del escount[:]
        del gtcount[:]

        with torch.no_grad():
            net.eval()
            time_stamp = 0.0
            for index,(timg,tden) in tqdm(enumerate(val_loader)):
                start = time.time()
                timg = timg.cuda()
                tden = tden.cuda()
                tes_den = net(timg)
                tloss = VALLOSS(tes_den,tden)

                valloss.update(tloss.item(),timg.shape[0])

                tes_count = np.sum(tes_den[0][0].cpu().detach().numpy())
                tgt_count = np.sum(tden[0][0].cpu().detach().numpy())

                escount.append(tes_count)
                gtcount.append(tgt_count)

                durantion = time.time()-start
                time_stamp+=durantion

                if index % 60 ==0 and epoch % 50 == 0 :


                    plt.subplot(131)
                    plt.title('raw image')
                    plt.imshow(timg[0][0].cpu().detach().numpy())

                    plt.subplot(132)
                    plt.title('gtcount:%.2f'%tgt_count)
                    plt.imshow(tden[0][0].cpu().detach().numpy())


                    plt.subplot(133)
                    plt.title('escount:%.2f'%tes_count)
                    plt.imshow(tes_den[0][0].cpu().detach().numpy())
                    plt.savefig(saveimg+'/epoch{}-step{}.jpg'.format(epoch, index))

                    # plt.show()
        valfps = len(val_loader)/time_stamp


        valmae, valmse = eva_model(escount, gtcount)



        info = 'valloss:{%.6f} @ valmae:{%.3f} @ valmse:{%.3f} @ valfps{%.3f}\n '%(
            valloss.avg * 10000,
            valmae,
            valmse,
            valfps
        )

        losssave =  {
            'epoch': epoch + 1,
            'state_dict':net.state_dict(),
            'best_loss': valloss.avg,
             'lr':get_learning_rate(optimizer)
        }


        if best_mae>valmae:
            best_mae = valmae
            torch.save(losssave, savemodel + '/best_loss_mae_' + method + '.pth')
        writer.add_scalars('data/loss', {
            'trainloss': trainloss.avg,
            'valloss': valloss.avg}, epoch)

        writer.add_scalars(method, {
            'valmse': valmse,
            'valmae': valmae,
            'trainmse': trainmse,
            'trainmae': trainmae
        }, epoch)
        logger.info(info)

    logger.info(method+' train complete')
    logger.info('best_loss:%.2f-- best_mae:%.2f -- best_mse:%.2f \n'%(best_loss*10000,best_mae,best_mse))
    logger.info('save bestmodel to '+savemodel)


