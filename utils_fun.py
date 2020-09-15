import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torch

import config


train_dataset = dataset.ImageFolder(config.train_dataset_path, transform=config.train_transform)
train_dataloader = DataLoader(train_dataset, 20, shuffle=True)
test_dataset = dataset.ImageFolder(config.test_dataset_path, transform=config.test_transform)
test_dataloader = DataLoader(test_dataset, 20, shuffle=False)


def get_features(data_loader, mode):
    # 定义是否使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features_list = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            net = torch.load(config.net_path).to(device)
            inputs, labels = data
            inputs = inputs.to(device)
            net.eval()
            _, features = net(inputs)
            features_list.append(features)
            labels_list.append(labels.numpy())
            print("extract the {} feature".format(i * 10))
            torch.cuda.empty_cache()

    features = torch.cat(features_list, dim=0).cpu().numpy()
    labels = np.concatenate(labels_list)
    if mode == 'train':
        np.save('train_features.npy', features, allow_pickle=True)
        np.save('train_labels.npy', labels, allow_pickle=True)
    if mode == 'test':
        np.save('test_features.npy', features, allow_pickle=True)
        np.save('test_labels.npy', labels, allow_pickle=True)
    return features, labels


def save_txt_files(path, the_list):
    if os.path.exists(path) is not True:
        f = open(path,'w')
        f.close()
    f = open(path, 'a')
    for i in the_list:
        f.write(str(i) + '\n')
    f.close()

def save_txt_files2(path, the_list):
    f = open(path, 'w')
    for i in the_list:
        f.write(str(i) + '\n')
    f.close()












