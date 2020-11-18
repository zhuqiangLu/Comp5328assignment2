import torch
from utils import one_hot_embedding, accuracy, AverageMeter
import torch.nn.functional as F
from network import FCNet #CONVNet
from data import get_loader
import torch.nn as nn
from estimator import Estimator, DT_Estimator
from loss import SCE, CE
def test(net, testloader, estimate_flip_rate=None, onGPU=True):

    top1_acc_meter_noise = AverageMeter()
    top1_acc_meter_clean = AverageMeter()
    top1_acc_meter_fr = AverageMeter()
    net.eval()

    count = 0 
    for _, data in enumerate(testloader, 0):
        feature, labels = data

        # convert to long for criterion
        labels = labels.long()
        if onGPU:
            feature = feature.cuda()
            labels = labels.cuda()
        

        min_batch_size = feature.size(0)
        nosie_preds, clean_preds  = net(feature)
        [top1_acc_noise] = accuracy(nosie_preds.data, labels.data, topk=(1,))

        #clean_preds = net.denoise(nosie_preds)

        #count += (clean_preds_tmp-clean_preds).mean()

        #print(tm.T.shape, softmax_preds.T.shape, preds.shape, clean_preds.shape)
        [top1_acc_clean] = accuracy(clean_preds.data, labels.data, topk=(1,))
        #[top1_acc_clean] = accuracy(clean_preds.data, labels.data, topk=(1,))



        top1_acc_meter_noise.update(top1_acc_noise.item(), min_batch_size)
        top1_acc_meter_clean.update(top1_acc_clean.item(), min_batch_size)

        if estimate_flip_rate is not None:
            estimate_flip_rate= estimate_flip_rate.float()
            estimate_clean = torch.mm(estimate_flip_rate.inverse(), nosie_preds.T).T
            [top1_acc_noise_est] = accuracy(estimate_clean.data, labels.data, topk=(1,))
            top1_acc_meter_fr.update(top1_acc_noise_est.item(), min_batch_size)
        else:
            top1_acc_meter_fr.update(top1_acc_clean.item(), min_batch_size)
    print(count)
    return top1_acc_meter_noise.avg, top1_acc_meter_clean.avg, top1_acc_meter_fr.avg


        


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
            

            nosie_preds, clean_preds  = net(feature)
        
            loss = criterion(nosie_preds, labels)
              
            # calculate accuracy
            [top1_acc] = accuracy(nosie_preds.data, labels.data, topk=(1,))
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
            estimator=None,
            epoch=10, 
            onGPU=True):

    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    

    prev_loss = float("inf")
    for i in range(epoch):
        net.train()

       
        with torch.enable_grad():
            for j, data in enumerate(trainloader, 0):

                feature, labels = data
                
                # convert to long for criterion
                labels = labels.long()
                if onGPU:
                    feature = feature.cuda()
                    labels = labels.cuda()
                
                
                optimizer.zero_grad()

                nosie_preds, clean_preds = net(feature)

                if estimator is not None:
                    estimator.update(nosie_preds, labels)

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

            

        val_acc, val_loss = val(net, valloader, criterion, onGPU=True)
        print("[{}] train loss: {:.3f}, val loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}".format(i,  loss_meter.avg, val_loss, top1_acc_meter.avg, val_acc))
                   
        # if val_loss - prev_loss> 0:
        #     break
        # prev_loss = val_loss

     
        



if __name__ == "__main__":

    
    transition_matrix = torch.Tensor(
        [[0.4, 0.3, 0.3],
         [0.3, 0.4, 0.3],
         [0.3, 0.3, 0.4]]
    )


    # transition_matrix = torch.Tensor(
    #     [[0.5, 0.2, 0.3],
    #      [0.3, 0.5, 0.2],
    #      [0.2, 0.3, 0.5]]
    # )

    
    estimator = DT_Estimator()

    trainloader, valloader, testloader = get_loader("../datasets/FashionMNIST0.6.npz", batch_size=256)
    net = FCNet(28*28, 3, transition_matrix).cuda()
    #net = CONVNet(1,3,transition_matrix).cuda()
    
    criterion = nn.CrossEntropyLoss()
    criterion = SCE()
    optimizer = torch.optim.SGD(
        params = net.parameters(),
        lr = 5e-4, 
        momentum = 0.9,
        weight_decay = 0.3
    )
    train(net,trainloader,valloader, optimizer, criterion, estimator=estimator, epoch=20)

    estimate_flip_rate = estimator.get_flip_rate()
    estimate_flip_rate = estimate_flip_rate.cuda()
    torch.save(net.state_dict(), "model.pth")
    #net.load_state_dict(torch.load("model.pth"))
    
    y_hat_accu, y_accu, est_y_acuu = test(net, testloader, estimate_flip_rate=estimate_flip_rate)
    print("before flip: {:.3f}, after flip: {:.3f}, after est flip {}".format(y_hat_accu, y_accu, est_y_acuu))

    
    print("estimate flip_rate: \n", estimate_flip_rate)

