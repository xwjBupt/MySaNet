
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr



def adjust_learning_rate(optimizer, epochs,epoch, LR = 0.001 ):
    lr = LR
    if epoch >= int(epochs * 0.8) :
      lr  = LR * 0.1
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    return lr
