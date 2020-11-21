import torch
import torch.nn as nn
import torch.nn.functional as F




class FCNet(nn.Module):

    def __init__(self, num_feature, num_class, flip_rate=None):
        super(FCNet, self).__init__()

        self.flip_rate = flip_rate
        if self.flip_rate is None:
            self.flip_rate = torch.eye(num_class)

        if flip_rate is not None:
            assert(num_class == flip_rate.shape[0] and num_class == flip_rate.shape[1])

        self.encode = nn.Sequential(
            nn.Linear(num_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # FC1 encoding layer
        self.FC1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # FC1 hidden branch 1
        self.FC1_2_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # FC1 hidden branch 2
        self.FC1_2_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # self.FC1_bottleneck = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        # )

       
        self.classifer = nn.Sequential(
            nn.Linear(1024, num_class),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.encode(x)
        
        #x_skip = x
        x = self.FC1(x)
        x_1 = self.FC1_2_1(x)
        x_2 = self.FC1_2_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        
        # use skip connection to avoid overfitting
        #x = self.FC1_bottleneck(x) 


        y = self.classifer(x)
        #print(y)
        # to generate noisy label using forward learning

       
            
        # if y.is_cuda:
        #     y_hat = torch.mm(self.flip_rate.cuda(), y.T).T
            
        # else:
        #     y_hat = torch.mm(self.flip_rate, y.T).T
        y_hat= y
        return y_hat, y

    def backward_learning(self, x):
        if self.flip_rate is not None:
            #softmax_preds = F.softmax(x, dim=0)
            if x.is_cuda:
                clean_preds = torch.mm(self.flip_rate.cuda().inverse(), x.T).T
            else:
                clean_preds = torch.mm(self.flip_rate.inverse(), x.T).T
            return clean_preds
        else:
            return x



