import numpy as np 
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, Dataset
import torch
from PIL import Image
from torchvision import transforms
class numpyDataset(Dataset):
    """  convert numpy array to PIL to tensor """

    def __init__(self, X, Y, transform=None):

        
        # reshape each as (c, w, h) 
        #self.X =  X.reshape(X.shape[0], -1, X.shape[-2], X.shape[-1])
        self.X = X
        
        self.Y = Y
        self.idxs = np.arange(X.shape[0])

        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[self.idxs[idx]]
        y = self.Y[self.idxs[idx]]
        if len(x.shape) == 3:
            x = Image.fromarray(np.uint8(x)).convert('RGB')
        else:
            x = Image.fromarray(np.uint8(x)).convert('L')

        if self.transform:
            x = self.transform(x)
        
        return x, y




def get_loader(path, transform, batch_size=4, split_ratio=0.8):

    # load numpy data
    dataset = np.load(path)

    # train  data 
    Xtr_val = dataset["Xtr"]
    Str_val = dataset["Str"]

   
    # test data
    Xts = dataset["Xts"]
    Yts = dataset["Yts"]


    train_dataset = numpyDataset(Xtr_val, Str_val, transform=transform)
    test_dataset = numpyDataset(Xts, Yts, transform=transform)

    # define train and val size
    train_size = int(len(train_dataset) * split_ratio)
    
    # then shuffle the indeces
    all_index = np.random.permutation(len(train_dataset))
    train_index = all_index[:train_size].tolist()
    val_index = all_index[train_size:].tolist()

    # define sampler
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)


    # get loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                                            num_workers=4, sampler=train_sampler)

    valloader = DataLoader(train_dataset, batch_size=batch_size,
                                            num_workers=4, sampler=val_sampler)

    testloader = DataLoader(test_dataset, batch_size=batch_size,
                                            num_workers=4, shuffle=True)

    return trainloader, valloader, testloader


def get_MNIST_05(transform=None, path=None, batch_size=128):

    # inject path
    if path is None:
        path = "../datasets/FashionMNIST0.5.npz"

    # inject transform
    if transform is None:

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
    

    transition_matrix = torch.Tensor(
        [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    )
    
    trainloader, valloader, testloader = get_loader(path, batch_size=batch_size, transform=transform)
    
    return trainloader, valloader, testloader, transition_matrix



def get_MNIST_06(transform=None, path=None, batch_size=128):

    # inject path
    if path is None:
        path = "../datasets/FashionMNIST0.6.npz"

    # inject transform
    if transform is None:

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
    
    transition_matrix = torch.Tensor(
        [[0.4, 0.3, 0.3],
         [0.3, 0.4, 0.3],
         [0.3, 0.3, 0.4]]
    )

    trainloader, valloader, testloader = get_loader(path, batch_size=batch_size, transform=transform)

    return trainloader, valloader, testloader, transition_matrix

def get_CIFAR(transform=None, path=None, batch_size=128):

    transition_matrix = None
    if path is None:
        path = "../datasets/CIFAR.npz"

    if transform is None:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainloader, valloader, testloader = get_loader(path, batch_size=batch_size, transform=transform)
    
    return trainloader, valloader, testloader, transition_matrix

