from train import test, val, train
from data import get_MNIST_06, get_MNIST_05, get_CIFAR
import torch
import torch.nn as nn
from estimator import Estimator, DT_Estimator
from loss import SCE, CE
from torchvision import transforms
from network import FCNet, Backbone
import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--loss', type=str,
                    default="ce", help='ce or sce')

parser.add_argument('--forward',  type=bool, 
                    default=False, help='enable forward T training')

parser.add_argument('--estimator', type=str,
                    default=None, help='Dual T or T?')

parser.add_argument('--given_flip_rate',type=bool,
                    default=False, help='Does it come with known flip rate?')

parser.add_argument('--dataset', type=str, 
                    default="mnist0.5", help='Does it come with known flip rate?')

parser.add_argument('--epoch',  type=int, default=20)

parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')

parser.add_argument('--weight_decay',type=float, 
                    default=0.3, help="regularizaiton weight")

parser.add_argument('--iters',type=int, 
                    default=10)



def train_by_arg(args):

    
    num_channel = 1
    if args.dataset == 'mnist05':
        print("dataset: mmnist0.5")
        trainloader, valloader, testloader, transition_matrix = get_MNIST_05()
        
    elif args.dataset == 'mnist06':
        print("dataset: mmnist0.6")
        trainloader, valloader, testloader, transition_matrix = get_MNIST_06()
       
    elif args.dataset == "cifar":
        print("dataset: cifar")
        trainloader, valloader, testloader, transition_matrix = get_CIFAR()
        num_channel=3
    else:
        print("dataset: mnist0.5 when input dataset is not recognise")
        trainloader, valloader, testloader, transition_matrix = get_MNIST_05()
    
    net = Backbone(num_channel, 3, None).cuda()

    if not args.given_flip_rate:
        print("no given flip rate")
        transition_matrix = None
    else:
        
        print("uses the given flip rate if applicable")

    if (args.forward):
        if transition_matrix is not None:
            print("enable forward training with this flip rate")
            net.flip_rate = transition_matrix
            print(transition_matrix)
        else:
            print("flip rate is none, using estimate T enable")
            if args.estimator == "DT":
                print("dual T estimation enable")
                estimator = DT_Estimator()
            elif args.estimator == "T":
                print("normal T estimation enable")
                estimator = Estimator()
            else:
                estimator = Estimator()
    
    if (args.loss == "ce"):
        print("loss: cross entropy")
        criterion = CE() #0.1,0.1
    else:
        print("loss: symemetric cross entropy")
        criterion = SCE(0.6, 0.2)


    if args.estimator == "DT":
        print("dual T estimation enable")
        estimator = DT_Estimator()
    elif args.estimator == "T":
        print("normal T estimation enable")
        estimator = Estimator()
    else:
        estimator = None

    if transition_matrix is not None and estimator is not None:
        print("as transition matrix is given, T estimation is disable")
        estimator = None
    

    
    optimizer = torch.optim.SGD(
        params = net.parameters(),
        lr = args.lr, 
        momentum = 0.9,
        weight_decay = args.weight_decay
    )
    
    train(net,trainloader,valloader, optimizer, criterion, estimator=estimator, epoch=args.epoch)

    if estimator is not None:
        
        T = estimator.get_flip_rate()
        print("the estimated flip rate is: \n", T)


    if args.forward and not args.given_flip_rate:
        
        net =  FCNet(28*28, 3, None).cuda()
        optimizer = torch.optim.SGD(
            params = net.parameters(),
            lr = args.lr, 
            momentum = 0.9,
            weight_decay = args.weight_decay
        )
        train(net,trainloader,valloader, optimizer, criterion, estimator=estimator, epoch=args.epoch)


    y_accu = test(net, testloader)
    print("test accuracy: {:.2f}".format(y_accu))
    return y_accu



if __name__ == "__main__":
    args = parser.parse_args()

    test_accu = []
    for i in range(args.iters):
        test_accu.append(train_by_arg(args))

    test_l = np.array(test_accu)
    print(test_l)
    print("mean : {}, std: {}".format(np.mean(test_l), np.std(test_l)))

    

