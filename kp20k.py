import torch.utils.data as data
import numpy as np
import pickle
import os


class KP20K(data.Dataset):
    def __init__(self, root, part, train=True):
        self.root = root
        self.train = train
        self.part = part

        if self.train:

            choose = {1: '1.npy', 2: '2.npy'}
            tr_d = np.load(
                os.path.join(root, choose[self.part]))
            self.train_data = tr_d[:, :470]
            self.train_labels = tr_d[:, 470:]

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
