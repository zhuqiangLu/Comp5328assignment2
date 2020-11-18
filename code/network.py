import torch
import torch.nn as nn
import torch.nn.functional as F




class FCNet(nn.Module):

    def __init__(self, num_feature, num_class, flip_rate=None):
        super(FCNet, self).__init__()

        self.flip_rate = flip_rate

        if flip_rate is not None:
            assert(num_class == flip_rate.shape[0] and num_class == flip_rate.shape[1])

        self.encode = nn.Sequential(
            nn.Linear(num_feature, 1024),
            nn.BatchNorm1d(1024),
        )

        # FC1 encoding layer
        self.FC1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # FC1 hidden branch 1
        self.FC1_2_1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        # FC1 hidden branch 2
        self.FC1_2_2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        self.FC1_bottleneck = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

       
        self.classifer = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class),
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
        x = self.FC1_bottleneck(x) 


        y = self.classifer(x)
        #print(y)
        # to generate noisy label using forward learning

        if self.flip_rate is not None:
            
            if y.is_cuda:
                y_hat = torch.mm(self.flip_rate.cuda(), y.T).T
            else:
                y_hat = torch.mm(self.flip_rate, y.T).T
        else:
            y_hat = None
        return y_hat, y

    def denoise(self, x):
        if self.flip_rate is not None:
            #softmax_preds = F.softmax(x, dim=0)
            if x.is_cuda:
                clean_preds = torch.mm(self.flip_rate.cuda().inverse(), x.T).T
            else:
                clean_preds = torch.mm(self.flip_rate.inverse(), x.T).T
            return clean_preds
        else:
            return x




# class CONVNet(nn.Module):

#     def __init__(self, num_feature, num_class, flip_rate=None):

#         super(CONVNet, self).__init__()
#         self.flip_rate = flip_rate

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(num_feature, 32, 5, dilation=3),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 128, 5, dilation=3),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, 3),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )


#         self.fc = nn.Sequential(
#             nn.Linear(1024, num_class),
#             #nn.Softmax(dim=1)
#         )

#     def forward(self, x):
        
#         #print(x.shape)
#         x = self.conv1(x)
#         #print(x.shape)
#         x = self.conv2(x)
#         #print(x.shape)
#         x = self.conv3(x)
#         #print(x.shape)
#         x = x.view(x.shape[0], -1)
#         #print(x.shape)
#         y = self.fc(x)


#          # to generate noisy label using forward learning
#         if y.is_cuda:
#             y_hat = torch.mm(self.flip_rate.cuda(), y.T).T
#         else:
#             y_hat = torch.mm(self.flip_rate, y.T).T
        
#         return y_hat, y


#     # def denoise(self, x):
#     #     if self.flip_rate is not None:
#     #         softmax_preds = F.softmax(x, dim=0)
#     #         if x.is_cuda:
#     #             clean_preds = torch.mm(self.flip_rate.cuda().inverse(), softmax_preds.T).T
#     #         else:
#     #             clean_preds = torch.mm(self.flip_rate.inverse(), softmax_preds.T).T
#     #         return clean_preds
#     #     else:
#     #         return x







if __name__ == "__main__":
    x = torch.rand((2, 3, 28,28))
    net = CONVNet(3,3)
    y = net(x)
    print(y)

