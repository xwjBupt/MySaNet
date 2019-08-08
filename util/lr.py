import torch

def get_learning_rate(optimizer):
    lr=[]
    if torch.cuda.device_count()>1:
        for param_group in optimizer.module.param_groups:
           lr +=[ param_group['lr'] ]

    else:
        for param_group in optimizer.param_groups:
            lr += [param_group['lr']]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr



def adjust_learning_rate(optimizer, epochs,epoch, LR = 0.001 ):

    if torch.cuda.device_count()>1:
        lr = LR
        if epoch >= int(epochs * 0.3) and epoch <= int(epochs * 0.5):
            lr = LR * 0.5
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = lr
        if epoch > int(epochs * 0.5) and epoch >= int(epochs * 0.8):
            lr = LR * 0.1
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = lr
        if epoch > int(epochs * 0.8):
            lr = LR * 0.01
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = lr

    else:
        lr = LR
        if epoch >= int(epochs * 0.3) and epoch<=int(epochs * 0.5) :
          lr  = LR * 0.5
          for param_group in optimizer.param_groups:
              param_group['lr'] = lr
        if epoch > int(epochs * 0.5) and epoch>=int (epochs * 0.8) :
            lr = LR * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch > int(epochs * 0.8) :
             lr = LR * 0.01
             for param_group in optimizer.param_groups:
                  param_group['lr'] = lr
    return lr
