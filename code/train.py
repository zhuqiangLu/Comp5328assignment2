import torch
from utils import one_hot_embedding, accuracy, AverageMeter
import torch.nn.functional as F
from network import FCNet
from data import get_loader
import torch.nn as nn

def test(net, testloader, tm, onGPU=True):

    top1_acc_meter_tm = AverageMeter()
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
        preds = net(feature.flatten(start_dim=1))
        [top1_acc] = accuracy(preds.data, labels.data, topk=(1,))

        softmax_preds = F.softmax(preds, dim=0)
        
        #clean_preds = torch.argmax(torch.mm(tm.T, softmax_preds.T), axis=0, keepdim=True).T
        clean_preds = torch.mm(tm.T, softmax_preds.T).T

        print(tm.T.shape, softmax_preds.T.shape, preds.shape, clean_preds.shape)
        [top1_acc_tm] = accuracy(clean_preds.data, labels.data, topk=(1,))

        top1_acc_meter.update(top1_acc.item(), min_batch_size)
        top1_acc_meter_tm.update(top1_acc_tm.item(), min_batch_size)

    return top1_acc_meter.avg, top1_acc_meter_tm.avg


        


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
            

            preds = net(feature.flatten(start_dim=1))
        
            loss = criterion(preds, labels)
              
            # calculate accuracy
            [top1_acc] = accuracy(preds.data, labels.data, topk=(1,))
            # record accuary and cross entropy losss
            min_batch_size = feature.size(0)
            top1_acc_meter.update(top1_acc.item(), min_batch_size)
            loss_meter.update(loss.item(), min_batch_size)
    
    return top1_acc_meter.avg, loss_meter.avg

def train(net, 
            trainloader, 
            valloader, 
            optimizer, 
            criterion, 
            scheduler=None, 
            epoch=10, 
            onGPU=True):

    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    for i in range(epoch):
        net.train()

        counter = 0
        with torch.enable_grad():
            for j, data in enumerate(trainloader, 0):

                feature, labels = data
                
                # convert to long for criterion
                labels = labels.long()
                if onGPU:
                    feature = feature.cuda()
                    labels = labels.cuda()
                
                
                optimizer.zero_grad()

                preds = net(feature.flatten(start_dim=1))
                loss = criterion(preds, labels)

                loss.backward()
                optimizer.step()
            
                # calculate accuracy
                [top1_acc] = accuracy(preds.data, labels.data, topk=(1,))
                # record accuary and cross entropy losss
                min_batch_size = feature.size(0)
                top1_acc_meter.update(top1_acc.item(), min_batch_size)
                loss_meter.update(loss.item(), min_batch_size)

                if j%100==99:
                    
                    val_acc, val_loss = val(net, valloader, criterion, onGPU=True)
                    print("[{}/{}train loss: {:.3f}, val loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}".format(i, counter, loss_meter.avg, val_loss, top1_acc_meter.avg, val_acc))
                    counter += 1

        if scheduler is not None:
            scheduler.step()



if __name__ == "__main__":

    print('loading data')
    trainloader, valloader, testloader = get_loader("../datasets/FashionMNIST0.5.npz", batch_size=128)
    net = FCNet(28*28, 3).cuda()
    print('loading net')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params = net.parameters(),
        lr = 1e-5, 
        momentum = 0.9,
        weight_decay = 0.1
    )
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)
    scheduler = None
    train(net,trainloader,valloader, optimizer, criterion, scheduler = scheduler, epoch=10)

    
    torch.save(net.state_dict(), "model.pth")
    #net.load_state_dict(torch.load("model.pth"))

    transition_matrix = torch.Tensor(
        [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    )
    y_hat_accu, y_accu = test(net, testloader,transition_matrix.cuda())
    print("before tm: {:.3f}, after tm{:.3f}".format(y_hat_accu, y_accu))

        

