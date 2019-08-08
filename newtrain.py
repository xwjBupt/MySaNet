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



    save = 50
    log_interval = 250
    lr = 0.000001
    epochs = 1000
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
    dataset = 'ShTARaw'
    method = 'can_720_zero'
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



    train_data = veichle(trainim_file,traingt_file,preload=True,phase = 'train')
    val_data = veichle(valim_file,valgt_file,preload=False,phase = 'val')

    train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=8)


    net = SANet(gray = False)


    if resume:
        cprint('=> loading checkpoint : ./rgb-randomcrop/model/modelbest_loss_cut_rgb-randomcrop.tar ',color='yellow')
        checkpoint = torch.load(current_dir +'/rgb-randomcrop/model/modelbest_loss_cut_rgb-randomcrop.tar')
        startepoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['state_dict'])
        # lr = checkpoint['lr']
        cprint("=> loaded checkpoint ",color='yellow')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    net.cuda()
    net.train()

    if finetune:
        # for i in [0, 2, 4]:
        #     for p in net.FME[i].parameters():
        #         p.requires_grad = False
        # cprint("=> freeze DME[0-4] grad ", color='yellow')
        for p in net.parameters():
            p.requires_grad = False

        for i in [15,16,17,18,19,20,21,22]:
            for p in net.DME[i].parameters():
                p.requires_grad = True
        cprint('=> fix all net,open last 3 conv',color='yellow')
        startepoch = 0

    logger.info(method)
    for epoch in range(0,epochs):
        trainmae = 0.
        trainmse = 0.
        valmae =0.
        valmse =0.
        trainloss = AverageMeter()
        valloss = AverageMeter()
        LR = adjust_learning_rate(optimizer, epochs, epoch, lr)
        logger.info('epoch:{} -- lr:{}'.format(epoch,LR))

        trainstart = time.time()
        step = 0.
        net.train()
        for index,(img,den) in tqdm(enumerate(train_loader)):
            step +=1
            img = img.cuda()
            den = den.cuda()

            es_den = net(img,den)

            loss = net.loss
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
        # writer.add_scalars('data/trainstate', {
        #                                   'trainmse': trainmse,
        #                                   'trainmae': trainmae}, epoch)

        info = 'trianloss:{%.6f} @ trainmae:{%.3f} @ trainmse:{%.3f} @ fps:{%.3f}'%(
                                                                                               trainloss.avg*10000,
                                                                                                trainmae, trainmse,
                                                                                               trainfps)

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
                tes_den = net(timg, tden)
                tloss = net.loss

                valloss.update(tloss.item(),timg.shape[0])

                tes_count = np.sum(tes_den[0][0].cpu().detach().numpy())
                tgt_count = np.sum(tden[0][0].cpu().detach().numpy())

                escount.append(tes_count)
                gtcount.append(tgt_count)

                durantion = time.time()-start
                time_stamp+=durantion
                if index % 60 ==0 and epoch % 200 == 0 :


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

        # if best_loss>valloss.avg:
        #     torch.save(losssave, savemodel+'/best_loss_cut_'+method+'.tar')
        #     best_loss = valloss.avg
        if best_mae>valmae:
            best_mae = valmae
            torch.save(losssave, savemodel + '/best_loss_mae_' + method + '.pth')
        # if best_mse>valmse:
        #     best_mse = valmse
        #     torch.save(losssave, savemodel + '/best_loss_mse_' + method + '.tar')

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


