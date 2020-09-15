import os
import torchvision.transforms as transforms

# dir
model_dir = 'Model_and_Log'
model_name = 'MRGCN_new_2'
model_path = 'model.pkl'

best_epoch = 'best_epoch'

model_dir_epoch = 'model_dir_epoch_lr'
model_dir_epoch_test = 'test'
model_dir_epoch_train = 'test'
model_test = 'model_test'

model_file = 'model.pt'
train_loss_file = 'train_loss.txt'
train_acc_file = 'train_acc.txt'
test_loss_file = 'test_loss.txt'
test_acc_file = 'test_acc.txt'

class_num = 6

# resnet
train_batch_size = 70
test_batch_size = 70


# GCN_model
features_dim_num = 2048
GCN_hidderlayer_dim_num = 512


# train_GCN
train_dataset_path = '1_multistage_malaria_classification/train'
test_dataset_path = '1_multistage_malaria_classification/test'
net_path = 'model-1/OurModel/113_84.0/net_best.pkl'

epoches = 500
k = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 注意与切割后的大小对应。不要出现小数
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])











