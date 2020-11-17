import torch
import torch.nn as nn




class FCNet(nn.Module):

    def __init__(self, num_feature, num_class):
        super(FCNet, self).__init__()

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

        self.FC2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.FC2_2_1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        self.FC2_2_2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        self.FC2_bottleneck = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )


        self.FC3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.FC3_2_1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        self.FC3_2_2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        self.FC3_bottleneck = nn.Sequential(
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
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.encode(x)
        
        x_skip = x
        x = self.FC1(x)
        x_1 = self.FC1_2_1(x)
        x_2 = self.FC1_2_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        
        # use skip connection to avoid overfitting
        x = self.FC1_bottleneck(x) + x_skip
        #print(x.shape)
        x_skip = x
        x = self.FC2(x)
        x_1 = self.FC2_2_1(x)
        x_2 = self.FC2_2_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.FC2_bottleneck(x) + x_skip

        x_skip = x
        x = self.FC3(x)
        x_1 = self.FC3_2_1(x)
        x_2 = self.FC3_2_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.FC3_bottleneck(x) + x_skip


        y = self.classifer(x)

        return y



if __name__ == "__main__":
    x = torch.rand((2, 100))
    net = FCNet(100, 10)
    y = net(x)
    print(y)

