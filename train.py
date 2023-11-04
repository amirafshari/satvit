from torch import optim
import torch.nn as nn
import torch

from torch.utils.data import DataLoader

from utils.dataset import PASTIS
from models.classification import Classification
from models.segmentation import Segmentation

import time
import yaml
import argparse


def train_seg(img_height=24, img_width=24, in_channel=10,
                patch_size=3, embed_dim=128, max_time=60,
                num_classes=20, num_head=8, dim_feedforward=2048,
                num_layers=8, batch_size=8,):


    data = PASTIS('./data/PASTIS/',)
    dataset = DataLoader(data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model
    model = Segmentation(img_width=24, img_height=24, in_channel=10, patch_size=3, embed_dim=128, max_time=60)
    model.to(device)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad == True])
    print(num_params)


    # Loss
    criterion = MaskedCrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    epoch = 10000
    model.train()

    for i in range(epoch):
        t1 = time.time()
        for img, label in dataset:
            img = img.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            output = model(img)

            # print(output[0], label[0])
            
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()


        if i % 10 == 0:
            torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f'epoch_{i}.pt')
        t2 = time.time()
        print('Epoch: ', i, 'Loss: ', loss)




def train_cls(img_height=9, img_width=9, in_channel=10,
                patch_size=3, embed_dim=128, max_time=60,
                num_classes=20, num_head=8, dim_feedforward=2048,
                num_layers=8, batch_size=32, model_type='classification',
                isResume=False, weight_path=None):
    # Dataset
    data = PASTIS('./data/PASTIS/', model_type=model_type)
    dataset = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model
    model = Classification(img_width=img_width, img_height=img_height, in_channel=in_channel, 
                           patch_size=patch_size, embed_dim=embed_dim, max_time=max_time, 
                           num_classes=num_classes, num_head=num_head, dim_feedforward=dim_feedforward, num_layers=num_layers)
    model.to(device)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad == True])
    print('Number of Parameters: ', num_params)


    # Loss
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)


    # Resume
    if isResume == True:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch = 0

    

    epochs = 10000
    model.train()

    for i in range(epochs):
        t1 = time.time()
        for img, label in dataset:
            img = img.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            output = model(img)

            # print(output[0], label[0])
            
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()


        if i % 10 == 0:
            torch.save({
                'epoch': i + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'./weights/epoch_{i + epoch}.pt')


        t2 = time.time()
        print('Epoch: ', i, 'Loss: ', loss, 'Time: ', (t2-t1)/60)






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Enter your parameters')
    
    parser.add_argument('--config_file', type=str, default="configs/classification.yaml",
                        help='.yaml configuration file to use')
    parser.add_argument('--weight', type=str, default='/weights/last.pt', action='store',
                        help='load pre-trained weight to resume training')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training')
    # parser.add_argument('--gpu_ids', type=list, default=["0", "1"], action='store',
    #                     help='gpu ids for running experiment')

    opt = parser.parse_args()


    config_file = opt.config_file
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)


    img_width = config['MODEL']['img_width']
    img_height = config['MODEL']['img_height']
    in_channel = config['MODEL']['in_channel']
    patch_size = config['MODEL']['patch_size']
    embed_dim = config['MODEL']['embed_dim']
    max_time = config['MODEL']['max_time']
    num_classes = config['MODEL']['num_classes']
    num_head = config['MODEL']['num_head']
    dim_feedforward = config['MODEL']['dim_feedforward']
    num_layers = config['MODEL']['num_layers']
    model = config['MODEL']['architecture']
    batch_size = config['TRAIN']['batch_size']

    isResume = opt.resume
    weight_path = opt.weight

    if model == 'classification':
        train_cls(img_width=img_width, img_height=img_height, in_channel=in_channel,
        patch_size=patch_size, embed_dim=embed_dim, max_time=max_time, num_classes=num_classes,
        num_head=num_head, dim_feedforward=dim_feedforward, num_layers=num_layers, batch_size=batch_size, isResume=isResume, weight_path=weight_path)

    elif model == 'segmentation':
        train_seg(img_width=img_width, img_height=img_height, in_channel=in_channel,
        patch_size=patch_size, embed_dim=embed_dim, max_time=max_time, num_classes=num_classes,
        num_head=num_head, dim_feedforward=dim_feedforward, num_layers=num_layers, batch_size=batch_size, isResume=isResume, weight_path=weight_path)


    # gpu_ids = opt.gpu_ids