import torch.nn.functional as F
import numpy as np
from utils import one_hot_embedding
import torch
class Estimator():

    def __init__(self):
        self.preds = None

    def update(self, preds, noise_label):

        
        if self.preds is None:
            self.preds = preds.cpu().data.numpy()
        else:
            self.preds = np.vstack((self.preds, preds.cpu().data.numpy()))

        
    def reset(self):
        self.preds = None

    def get_flip_rate(self):

        num_class = self.preds.shape[-1]

        flip_rate = np.zeros((num_class, num_class))
        #print(num_class)
        for i in range(num_class):
            flip_rate[:, i] = self.preds[self.preds[:, i].argsort()][-1].T
        
        return torch.from_numpy(flip_rate).float()




class DT_Estimator():

    def __init__(self):
        self.preds = None
        self.noise_label = None

    def update(self, preds, noise_label):

        
        if self.preds is None:
            self.preds = F.softmax(preds, dim=1).cpu().data.numpy()
        else:
            self.preds = np.vstack((self.preds, F.softmax(preds, dim=1).cpu().data.numpy()))

        if self.noise_label is None:
            self.noise_label =noise_label.cpu().data.numpy()
        else:
            self.noise_label = np.hstack((self.noise_label, noise_label.cpu().data.numpy()))


        
    def reset(self):
        self.preds = None

    def _get_intermediate_T(self):

        num_class = self.preds.shape[-1]

        flip_rate = np.zeros((num_class, num_class))
        #print(num_class)
        for i in range(num_class):
            flip_rate[:, i] = self.preds[self.preds[:, i].argsort()][-1].T
        
        return flip_rate

       
    def get_flip_rate(self):
        num_class = self.preds.shape[-1]

        T_i = torch.from_numpy(self._get_intermediate_T()).float()
        preds = torch.from_numpy(self.preds).float()

        noise_label = torch.from_numpy(self.noise_label)
        y_prime = torch.mm(T_i.inverse(), preds.T).T
        T_shade = torch.zeros_like(T_i)

        #noise_one_hot_label = one_hot_embedding(self.noise_label, num_class)

        # temp = one_hot_embedding( np.argmax(self.preds, axis=1), num_class) 
        # num = temp * noise_one_hot_label

        # num = torch.sum(num, dim=0).reshape((1, num_class))
        
        # den = torch.sum(temp, dim=0).reshape((num_class, 1))
       

        for l in range(num_class):
            # get indeces that argmax is l

            argmax_l = torch.argmax(y_prime, dim=1)
            idx_l = torch.nonzero(argmax_l == l,as_tuple=True)


            for j in range(num_class):
           
                
                idxs = torch.nonzero(noise_label[idx_l] == j, as_tuple=True)
                num = len(idxs[0])
        

                T_shade[l][j] = num/(len(idx_l[0]))

                

        return T_shade




