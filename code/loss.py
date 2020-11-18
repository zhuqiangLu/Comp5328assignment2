import torch
import torch.nn as nn

import torch.nn.functional as F
from utils import sigmoid_loss_labels_embedding, one_hot_embedding

class SCE(nn.Module):

    def __init__(self):

        super(SCE, self).__init__()

    def forward(self, x, y):
        y = one_hot_embedding(y, x.shape[-1])

        n = x.shape[0] * x.shape[-1]
        
        if x.is_cuda:
            y = y.cuda()
        #CE
        L_ce = -(torch.sum(y * torch.log(x), dim=0))/n
        # RCE
        L_rce = -(torch.sum(x * torch.log(y+1e-7), dim= 0))/n

        
        return (L_ce + L_rce).mean()



class CE(nn.Module):

    def __init__(self):

        super(CE, self).__init__()

    def forward(self, x, y):
        y = one_hot_embedding(y, x.shape[-1])

        n = x.shape[0] * x.shape[-1]
        #CE
        if x.is_cuda:
            y = y.cuda()
        L_ce = -(torch.sum(y *torch.log(x), dim=0)/x.shape[0])
        

        
        return L_ce.sum()/x.shape[-1]

