import torch.utils.data as data
import numpy as np
import pickle
import os


class KP20K(data.Dataset):
    def __init__(self, root, part, scale, train=True):
        self.root = root
        self.train = train
        self.part = part
        self.scale = scale
        

        if self.train:
           
            tr_d = np.load(
                    os.path.join(root, '1.npy'))
                
            tr_d_ =  np.load(
                    os.path.join(root, '2.npy'))
                
            tr_d = np.concatenate((tr_d,trd_),axis=0)
                
            a= np.random.randint( 0,high = 1000000, size = 1) 
            a=a[0]
            scale_select={'large':1000000, 'mid': 500000, 'small': 250000}
            self.train_data = tr_d[a[0]:a[0+]scale_select[self.scale], :470]
            self.train_labels = tr_d[:scale_select[self.scale], 470:]

            '''
            if self.part == 1:
                tr_d = np.load(
                    os.path.join(root, '1.npy'))
                self.train_data = tr_d[:, :470]
                self.train_labels = tr_d[:, 470:]
            
            else:
                tr_d = np.load(
                    os.path.join(root, '2.npy'))
                self.train_data = tr_d[:, :470]
                self.train_labels = tr_d[:, 470:]
        '''
        else:
            with open(os.path.join(root, 'test_data.pkl'), 'rb') as f1:
                te_d = pickle.load(f1)
            with open(os.path.join(root, 'test_label.pkl'), 'rb') as f2:
                te_l = pickle.load(f2)
            self.test_data = te_d
            self.test_labels = te_l

    def __getitem__(self, index):
        if self.train:
            input_sequence, target = self.train_data[index], self.train_labels[index]
            return input_sequence, target
        else:
            input_sequence, target = self.test_data[index], self.test_labels[index]
            return input_sequence, target

    def __len__(self):
        if self.train:
            # length = {1: 1133888, 2: 1131628}
            return len(self.train_data)
        else:
            return len(self.test_data)

