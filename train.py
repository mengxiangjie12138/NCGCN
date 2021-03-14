import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import time
from ResNet import ResNet50
import config
from torchsampler.imbalanced import ImbalancedDatasetSampler


def train(load_model=False):
    train_dataset = dataset.ImageFolder(config.train_path, transform=config.train_transform)
    train_data_loader = DataLoader(train_dataset, config.source_batch_size, sampler=ImbalancedDatasetSampler(train_dataset))
    test_dataset = dataset.ImageFolder(config.train_path, transform=config.test_transform)
    test_data_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=True)

    test_dataset = dataset.ImageFolder(config.test_path, transform=config.test_transform)
    test_data_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=False)

    # define GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    net = ResNet50(num_classes=config.class_num).to(device)
    if load_model:
        net = torch.load(config.model_path)

    cross_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)

    for epoch in range(config.epoches):
        sum_loss = 0.
        correct = 0.
        total = 0.
        since = time.time()
        net.train()
        length = config.source_batch_size + config.target_batch_size
        dis_list = []

        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, dis = net(inputs, labels=labels)
            dis_list.append(dis)
            loss1 = cross_loss(outputs, labels)

            loss = loss1
            sum_loss += loss1

            _, pre = torch.max(outputs.data, 1)

            total += outputs.size(0)
            correct += torch.sum(pre == labels.data)
            train_acc = correct / total

            loss.backward()
            optimizer.step()

            iter_num = i + 1 + epoch * length
            print('[epoch:%d, iter:%d] Loss: %f | Train_acc: %f | Time: %f'
                  % (epoch + 1, iter_num, sum_loss / i, train_acc, time.time() - since))

        mean_dis = np.mean(dis_list)

        # start to test
        if epoch % 1 == 0:
            print("start to test:")
            with torch.no_grad():
                correct = 0.
                total = 0.
                loss = 0.
                for i, data in enumerate(test_data_loader):
                    net.eval()
                    inputs_test, labels_test = data
                    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                    outputs_test, _ = net(inputs_test, mean_dis)
                    loss += cross_loss(outputs_test, labels_test)

                    # present_max, pred = torch.max(outputs.data, 1)
                    _, pred = torch.max(outputs_test.data, 1)

                    total += labels_test.size(0)
                    correct += torch.sum(pred == labels_test.data)
                test_acc = correct / total
                print('test_acc:', test_acc, '| time', time.time() - since)


if __name__ == '__main__':
    train()
