import logging
from termcolor import cprint
import sys
import math


logger = logging.getLogger(name='lib')
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

def eva_model(escount,gtcount):
        mae = 0.
        mse = 0.
        for i in range(len(escount)):
            temp1 = abs(escount[i]-gtcount[i])
            temp2 = temp1*temp1
            mae += temp1
            mse += temp2
        MAE = mae*1./len(escount)
        MSE = math.sqrt(1./len(escount)*mse)
        return MAE,MSE




