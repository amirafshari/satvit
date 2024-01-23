import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
import torch.nn.functional as F
import os
from copy import deepcopy
import pandas as pd
import numpy as np




class CutOrPad(object):
    """
    Pad series with zeros (matching series elements) to a max sequence length or cut sequential parts
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels, seq_lengths

    REMOVE DEEPCOPY OR REPLACE WITH TORCH FUN
    """

    def __init__(self, max_seq_len=60, random_sample=False, from_start=False):
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample
        self.from_start = from_start
        assert int(random_sample) * int(from_start) == 0, "choose either one of random, from start sequence cut methods but not both"

    def __call__(self, sample):
        seq_len = deepcopy(sample['img'].shape[0])
        sample['img'] = self.pad_or_cut(sample['img'])
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        sample['seq_lengths'] = seq_len
        return sample

    def pad_or_cut(self, tensor, dtype=torch.float32):
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        if diff > 0:
            tsize = list(tensor.shape)
            if len(tsize) == 1:
                pad_shape = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
            tensor = torch.cat((tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
        elif diff < 0:
            if self.random_sample:
                return tensor[self.random_subseq(seq_len)]
            elif self.from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
            tensor = tensor[start_idx:start_idx+self.max_seq_len]
        return tensor
    
    def random_subseq(self, seq_len):
        return torch.randperm(seq_len)[:self.max_seq_len].sort()[0]



def get_rgb(x, batch_index=0, t_show=1):
    """Utility function to get a displayable rgb image 
    from a Sentinel-2 time series.
    """
    im = x[t_show, [2,1,0]]
    mx = im.max(axis=(1,2))
    mi = im.min(axis=(1,2))   
    im = (im - mi[:,None,None])/(mx - mi)[:,None,None]
    im = im.swapaxes(0,2).swapaxes(0,1)
    im = np.clip(im, a_max=1, a_min=0)
    return im



class PASTIS(Dataset):
    def __init__(self, pastis_path):
        self.pastis_path = pastis_path
        self.file_names = os.listdir(self.pastis_path)
        # self.length = len(self.file_names)//10
        # random.shuffle(self.file_names)
        # self.file_names = self.file_names[:self.length]
        self.to_cutorpad = CutOrPad()

    def __len__(self):
        return len(self.file_names)

    def add_date_channel(self, img, doy):
        img = torch.cat((img, doy), dim=1)
        return img

    def normalize(self, img):
        C = img.shape[1]
        mean = img.mean(dim=(0, 2, 3)).to(torch.float32).reshape(1, C, 1, 1)
        std = img.std(dim=(0, 2, 3)).to(torch.float32).reshape(1, C, 1, 1)
        img = (img - mean) / std
        return img

    def __getitem__(self, idx):
        data = pd.read_pickle(os.path.join(self.pastis_path, self.file_names[idx]))
        
        data['img'] = data['img'].astype('float32')
        data['img'] = torch.tensor(data['img'])
        data['img'] = self.normalize(data['img'])
        T, C, H, W = data['img'].shape

        data['labels'] = data['labels'].astype('long')
        data['labels'] = torch.tensor(data['labels'])

        data['doy'] = data['doy'].astype('float32')
        data['doy'] = torch.tensor(data['doy'])
        data['doy'] = data['doy'].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        data['doy'] = data['doy'].repeat(1, 1, H, W)

        data['img'] = self.add_date_channel(data['img'], data['doy'])
        del data['doy']

        data = self.to_cutorpad(data)
        del data['seq_lengths']

        return data['img'], data['labels']

class PASTISDataLoader:
    def __init__(self, dataset_path, batch_size=16, train_ratio=0.7, val_ratio=0.15):
        self.dataset = PASTIS(dataset_path)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def get_data_loaders(self):
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def print_sample_data(self):
        files = os.listdir(self.dataset.pastis_path)
        file = random.choice(files)
        data = pd.read_pickle(os.path.join(self.dataset.pastis_path, file))
        print(data.keys())
        print('Image: ', data['img'].shape)
        print('Labels: ', data['labels'].shape, data['labels'])
        print('DOY: ', data['doy'].shape, data['doy'])