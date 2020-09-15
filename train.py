import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import os
import time
import utils_fun
from ResNet2 import ResNet50
import config
from torchsampler import ImbalancedDatasetSampler

# use visual GPU
os.environ['CUDA_VISIBLE_DEVICES']='1'


def train(k=10):
    # make each epoch train and test directory
    dir_list = ['predicts_epoch_k','lables_epoch_k','features_epoch_k','present_epoch_k','model_epoch_k',]
    if os.path.join(config.model_dir_epoch,'k={}'.format(k)) is not True:
        for i in dir_list:
            os.makedirs(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test',i+'={}'.format(k)))
        for i in dir_list:
            os.makedirs(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train',i+'={}'.format(k)))

    if os.path.join(config.best_epoch, 'k={}'.format(k)) is not True:
        os.makedirs(os.path.join(config.best_epoch, 'k={}'.format(k)))

    train_dataset = dataset.ImageFolder(config.train_dataset_path, transform=config.train_transform)
    train_dataloader = DataLoader(train_dataset, config.train_batch_size, sampler=ImbalancedDatasetSampler(train_dataset))
    test_dataset = dataset.ImageFolder(config.test_dataset_path, transform=config.test_transform)
    test_dataloader = DataLoader(test_dataset, config.test_batch_size, shuffle=False)

    # define GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    if os.path.exists(config.model_path) is not True:
        net = ResNet50(num_classes=config.class_num).to(device)
    else:
        net = torch.load(config.model_path)

    cross_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    best_acc = 20

    if os.path.exists(config.model_path) is not True:
        for epoch in range(config.epoches):
            # mei ge epoch zhi ling
            train_labels_list = []
            train_predicted_list = []
            train_present_list = []
            train_features_list = []

            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            length = len(train_dataloader)

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs, features,_ = net(inputs, k) # bu shi torch bian liang
                loss = cross_loss(outputs, labels)

                loss.backward()
                optimizer.step()
                sum_loss += loss.item() # item转换tensor

                _, pre = torch.max(outputs.data, 1)
                present_all = outputs.data.cpu().numpy() # yi jing zhaun wei numpy

                total += labels.size(0)
                correct += torch.sum(pre == labels.data)
                acc = 100. * correct / total


                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%%'
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

                x = sum_loss / (i + 1)

                for i in pre:
                   train_predicted_list.append(i.item())
                for i in present_all:
                    train_present_list.append(i)
                for i in labels.data:
                   train_labels_list.append(i.item())
                for i in features:
                    train_features_list.append(i)

            train_loss_list.append(x)
            y = (100. * correct / total).item()
            train_acc_list.append(y)

            # save accuracy file for txt
            utils_fun.save_txt_files2(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','loss.txt'),train_loss_list)
            utils_fun.save_txt_files2(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','acc.txt'),train_acc_list)

            scheduler.step(epoch)

            # start to test
            if epoch % 1 == 0:
                print("start to test:")
                with torch.no_grad(): 
                    correct = 0
                    correct1 = 0
                    total = 0
                    loss = 0.0 

                    features_list = []
                    labels_list_num = []
                    predicted_list_num = []
                    present_all_list = []
                    for i, data in enumerate(test_dataloader):
                        net.eval()
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs, features, _ = net(inputs, k)
                        loss += cross_loss(outputs, labels)

                        features=torch.from_numpy(features) # convert features to ndarray



                       #  present_max, pred = torch.max(outputs.data, 1) 
                        _, pred = torch.max(outputs.data, 1) 
                        present_all = outputs.data.cpu().numpy() 
                        # _, pred1 = torch.max(res_outputs.data, 1) # resnet predict



                        total += labels.size(0)
                        correct += torch.sum(pred == labels.data)
                        # correct1 += torch.sum(pred1 == labels.data) # resnet correct

                        acc = 100. * correct / total 
                        # acc1 = 100. * correct1 / total # resnet acc1

                        for i in pred:
                            predicted_list_num.append(i.item())
                        for i in present_all:
                            present_all_list.append(i)
                        for i in labels.data:
                            labels_list_num.append(i.item())
                        for i in features.cpu().data.numpy():
                            features_list.append(i)

                    test_loss_list.append(loss.item())
                    test_acc = 100. * correct / total
                    test_acc_list.append(test_acc.item())

                    # save test acc for txt
                    utils_fun.save_txt_files2(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','loss.txt'),
                                              test_loss_list)
                    utils_fun.save_txt_files2(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','acc.txt'),
                                              test_acc_list)

                    # save the best epoch
                    if acc>=best_acc:
                        best_acc=acc
                        print('the best epoch+1 updating{}'.format(epoch+1))

                        # write best epoch number
                        f = open(os.path.join(config.model_dir_epoch,'k={}'.format(k), 'best_epoh.txt'), 'w')
                        f.write(str(epoch+1))
                        f.close()



                        # save for np
                        np.save(os.path.join(config.best_epoch,'features.npy'), features_list)
                        np.save(os.path.join(config.best_epoch, 'predict_num.npy'),
                                predicted_list_num)
                        np.save(os.path.join(config.best_epoch, 'lables_num.npy'),
                                labels_list_num)
                        np.save(os.path.join(config.best_epoch,'present_all.npy'), present_all_list)
                        # save predict and labels for txt
                        utils_fun.save_txt_files2(os.path.join(config.best_epoch,'predict_num_t.txt'), predicted_list_num)
                        utils_fun.save_txt_files2(os.path.join(config.best_epoch,'lables_num_t.txt'), labels_list_num)



                        # save best epoch train
                        if os.path.join(config.best_epoch,'k={}'.format(k),'train') is not True:
                            os.makedirs(os.path.join(config.best_epoch,'train'))
                        np.save(os.path.join(config.best_epoch,'train','features.npy'),train_features_list)
                        np.save(os.path.join(config.best_epoch,'train','present.npy'),train_present_list)
                        np.save(os.path.join(config.best_epoch,'train','predict.npy'),train_predicted_list)
                        np.save(os.path.join(config.best_epoch,'train','label.npy'),train_labels_list)

                        # save the best model
                        model_path = os.path.join(config.best_epoch,'k={}'.format(k))
                        if os.path.exists(model_path) is not True:
                            os.makedirs(model_path)
                        else:
                            torch.save(net, os.path.join(model_path + 'model.pkl_{}'.format(best_acc)))


                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','predicts_epoch_k={}'.format(k), 'epoch{}_acc{}_predict.npy'.format(epoch, acc)), predicted_list_num)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','lables_epoch_k={}'.format(k), 'epoch{}_acc{}_lables.npy'.format(epoch, acc)), labels_list_num)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','features_epoch_k={}'.format(k), 'epoch{}_acc{}_feature.npy'.format(epoch, acc)), features_list)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'test','present_epoch_k={}'.format(k), 'epoch{}_acc{}_present.npy'.format(epoch, acc)), present_all_list)
                    # torch.save(net, os.path.join(config.model_dir_epoch_lr, 'model_epoch_k={}'.format(k), 'epoch{}_acc{}_model.pkl'.format(epoch, acc)))

                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','features_epoch_k={}'.format(k), 'epoch{}_acc{}_features.npy'.format(epoch, acc)), train_features_list)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','present_epoch_k={}'.format(k), 'epoch{}_acc{}_present.npy'.format(epoch, acc)), train_present_list)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','predicts_epoch_k={}'.format(k), 'epoch{}_acc{}_predict.npy'.format(epoch, acc)), train_predicted_list)
                    np.save(os.path.join(config.model_dir_epoch,'k={}'.format(k),'train','lables_epoch_k={}'.format(k), 'epoch{}_acc{}_label.npy'.format(epoch, acc)), train_labels_list)


                    print('the test acc is:{:.4f}%, the loss is:{}'.format(acc, loss))

    else:
        test_loss_list = []
        test_acc_list = []

        print("start to test:")
        with torch.no_grad():  
            correct = 0
            correct1 = 0
            total = 0
            loss = 0.0  

            features_list = []
            labels_list_num = []
            predicted_list_num = []
            present_all_list = []
            for i, data in enumerate(test_dataloader):
                net.eval()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, features, _ = net(inputs, k)
                loss += cross_loss(outputs, labels)

                features = torch.from_numpy(features)  # convert features to ndarray

                #  present_max, pred = torch.max(outputs.data, 1) 
                _, pred = torch.max(outputs.data, 1)  
                present_all = outputs.data.cpu().numpy()  
                # _, pred1 = torch.max(res_outputs.data, 1) # resnet predict

                total += labels.size(0)
                correct += torch.sum(pred == labels.data)
                # correct1 += torch.sum(pred1 == labels.data) # resnet correct

                acc = 100. * correct / total  # acc为什么还会存在
                # acc1 = 100. * correct1 / total # resnet acc1

                for i in pred:
                    predicted_list_num.append(i.item())
                for i in present_all:
                    present_all_list.append(i)
                for i in labels.data:
                    labels_list_num.append(i.item())
                for i in features.cpu().data.numpy():
                    features_list.append(i)

            test_loss_list.append(loss.item())
            test_acc = 100. * correct / total
            test_acc_list.append(test_acc.item())

            if os.path.exists(config.model_test) is not True:
                os.makedirs(config.model_test)

            # save test acc for txt
            utils_fun.save_txt_files2(os.path.join(config.model_test,'loss.txt'),
                                      test_loss_list)
            utils_fun.save_txt_files2(os.path.join(config.model_test,'acc.txt'),
                                      test_acc_list)
            np.save(os.path.join(config.model_test,
                    'epoch_acc{}_predict.npy'.format(acc)), predicted_list_num)
            np.save(os.path.join(config.model_test,
                    'epoch_acc{}_labels.npy'.format(acc)), labels_list_num)
            np.save(os.path.join(config.model_test,
                    'epoch_acc{}_features.npy'.format(acc).format(acc)), features_list)
            np.save(os.path.join(config.model_test,
                    'epoch_acc{}_present.npy'.format(acc).format(acc)), present_all_list)

            print('the test acc is:{:.4f}%, the loss is:{}'.format(acc, loss))

if __name__ == '__main__':
    train()








































