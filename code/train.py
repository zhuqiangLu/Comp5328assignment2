import torch
from utils import one_hot_embedding, accuracy, AverageMeter
import torch.nn.functional as F
from network import FCNet #CONVNet
from data import get_loader
import torch.nn as nn
from estimator import Estimator, DT_Estimator
from loss import SCE, CE
from torchvision import transforms

def test(net, testloader, estimate_flip_rate=None, onGPU=True):

    top1_acc_meter = AverageMeter()
    net.eval()

    for _, data in enumerate(testloader, 0):
        feature, labels = data

        # convert to long for criterion
        labels = labels.long()
        if onGPU:
            feature = feature.cuda()
            labels = labels.cuda()
        

        min_batch_size = feature.size(0)
        _, clean_preds  = net(feature)
        [top1_acc] = accuracy(clean_preds.data, labels.data, topk=(1,))

        top1_acc_meter.update(top1_acc.item(), min_batch_size)

    return top1_acc_meter.avg


        


def val(net, valloader, criterion, onGPU=True):

    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.eval()
    with torch.no_grad():
        

        for _, data in enumerate(valloader, 0):
            # get the inputs
            feature, labels = data
            # convert to long for criterion
            labels = labels.long()
            if onGPU:
                feature = feature.cuda()
                labels = labels.cuda()
            

            nosie_preds, _  = net(feature)
        
            loss = criterion(nosie_preds, labels)
              
            # calculate accuracy
            [top1_acc] = accuracy(nosie_preds.data, labels.data, topk=(1,))
            # record accuary and cross entropy losss
            min_batch_size = feature.size(0)
            top1_acc_meter.update(top1_acc.item(), min_batch_size)
            loss_meter.update(loss.item(), min_batch_size)
    
    return top1_acc_meter.avg, loss_meter.avg



def train(net, trainloader, valloader, optimizer, criterion, estimator=None, epoch=10, onGPU=True):


    


    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    prev_loss = float("inf")
    for i in range(epoch):
        net.train()

       
        with torch.enable_grad():
            for _, data in enumerate(trainloader, 0):

                feature, labels = data
                
                # convert to long for criterion
                labels = labels.long()
                if onGPU:
                    feature = feature.cuda()
                    labels = labels.cuda()
                
                
                optimizer.zero_grad()

                nosie_preds, _ = net(feature)

                loss = criterion(nosie_preds, labels)
                #print(loss.shape)
                loss.backward()
                optimizer.step()
            
                # calculate accuracy
                [top1_acc] = accuracy(nosie_preds.data, labels.data, topk=(1,))

                # record accuary and cross entropy losss
                min_batch_size = feature.size(0)
                top1_acc_meter.update(top1_acc.item(), min_batch_size)
                loss_meter.update(loss.item(), min_batch_size)
        

        val_acc, val_loss = val(net, valloader, criterion, onGPU=onGPU)
        print("[{}] train loss: {:.3f}, val loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}".format(i+1,  loss_meter.avg, val_loss, top1_acc_meter.avg, val_acc))
        ## estimate
        if estimator is not None and i == int(epoch/2):
            print("Estimating flip rate")

            net.eval()
            with torch.no_grad():
                for _, data in enumerate(valloader, 0):
                    feature, labels = data
                        
                    # convert to long for criterion
                    labels = labels.long()
                    if onGPU:
                        feature = feature.cuda()
                        labels = labels.cuda()
                    
                    nosie_preds, _ = net(feature)

                    estimator.update(nosie_preds, labels)

            


     







 
