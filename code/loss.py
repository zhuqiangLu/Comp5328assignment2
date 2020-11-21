import torch
import torch.nn as nn

import torch.nn.functional as F
from utils import sigmoid_loss_labels_embedding, one_hot_embedding

class SCE(nn.Module):

    def __init__(self, alpha, beta):

        super(SCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_gt):
        y_gt = one_hot_embedding(y_gt, y_pred.shape[-1])

        n = y_pred.shape[0] 
        
        if y_pred.is_cuda:
            y_gt = y_gt.cuda()
        #CE
        L_ce =   -(torch.sum(y_gt * torch.log(y_pred), dim=0))/n
        L_ce = self.alpha * ((torch.sum(L_ce))/y_pred.shape[1])
        # RCE
        L_rce = self.beta * -(torch.sum(y_pred * torch.log(y_gt+1e-7), dim= 0))/n
        L_rce = self.beta * (torch.sum(L_rce)/y_pred.shape[1])

        
        return (L_ce + L_rce)



class CE(nn.Module):

    def __init__(self):

        super(CE, self).__init__()

    def forward(self, y_pred, y_gt):
        y_gt = one_hot_embedding(y_gt, y_pred.shape[-1])

        #CE
        if y_pred.is_cuda:
            y_gt = y_gt.cuda()
        L_ce = -(torch.sum(y_gt * torch.log(y_pred), dim=0)/y_pred.shape[0])
        

        
        return L_ce.sum()/y_pred.shape[-1]

