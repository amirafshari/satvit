# pastis_module.py
import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from utils.dataset import CutOrPad, get_rgb

class PASTIS(Dataset):
    def __init__(self, pastis_path):
        self.pastis_path = pastis_path
        self.file_names = os.listdir(self.pastis_path)
        random.shuffle(self.file_names)
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


