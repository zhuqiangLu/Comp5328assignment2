import numpy as np 
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import torch



def get_loader(path, batch_size=4, split_ratio=0.8):

    # load numpy data
    dataset = np.load(path)

    # train  data 
    Xtr_val = dataset["Xtr"]
    Str_val = dataset["Str"]

   
    # test data
    Xts = dataset["Xts"]
    Yts = dataset["Yts"]

    # reshape each as (c, w, h) 
    Xtr_val = Xtr_val.reshape(Xtr_val.shape[0], -1, Xtr_val.shape[-2], Xtr_val.shape[-1])
    Xts = Xts.reshape(Xts.shape[0], -1, Xts.shape[-2], Xts.shape[-1])
    # print(Xtr_val.shape)
    # print(Str_val.shape)
    # print(Xts.shape)
    # print(Yts.shape)

    # wrap as torch tensor
    train_feature_tensor = torch.Tensor(Xtr_val)
    train_label_tensor = torch.Tensor(Str_val)
    test_feature_tensor = torch.Tensor(Xts)
    test_label_tensor = torch.Tensor(Yts)

    # build dataset 
    train_dataset = TensorDataset(train_feature_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_feature_tensor, test_label_tensor)

    # split train dataset

    # define train and val size
    all_index = np.random.permutation(len(train_dataset))
    train_size = int(len(train_dataset) * split_ratio)
    val_size = int(len(train_dataset) * (1 - split_ratio))

    # then shuffle the indeces
    idxs = np.random.permutation(len(train_dataset))
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

if __name__ == "__main__":
    trainloader, valloader, testloader = get_loader("../datasets/FashionMNIST0.5.npz")
    for i, data in enumerate(trainloader):
        print(data[1])
        break