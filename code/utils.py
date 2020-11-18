import numpy as np
import torch


def sigmoid_loss_labels_embedding(labels, num_classes):

    base = -1 * torch.ones((labels.shape[0], num_classes))
    
    one_hot = 2 * one_hot_embedding(labels, num_classes)

    return base + one_hot



# Embedding labels to one-hot form.
def one_hot_embedding(labels, num_classes):
    r"""Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 


# Calcuate the accuracy according to the prediction and the true label.
def accuracy(output, target, topk=(1,)):
    r"""Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count



if __name__ == "__main__":
    
    y = torch.Tensor([2,0,1,1,0]).long()
    print(sigmoid_loss_labels_embedding(y, 3))