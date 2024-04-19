'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import sys
sys.path.insert(0, './')
import torch
import torch.nn as nn
import torch.optim as optim

from util.data_parser import parse_data
from util.model_parser import parse_model
from util.utility import find_gpu, Logger

from util.model_utils import train_model, test_model


# 使用argparse处理命令行参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# dataset，默认值为'cifar10'
parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'The dataset used, default = "cifar10".')
# batch_size，用于指定批量大小
parser.add_argument('--batch_size', type = int, default = 128, help = 'The batch size, default is 128.')
# valid_ratio，用于指定验证集的比例
parser.add_argument('--valid_ratio', type = float, default = None, help = 'The proportion of the validation set, default is None.')

# model_type，用于指定模型类型
parser.add_argument('--model_type', type = str, default = 'resnet18', help = 'The type of the model, default is "resnet18".')
# normalize，用于指定标准化模式
parser.add_argument('--normalize', type = str, default = None, help = 'The nomralization mode, default is None.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


# 设备选择
device = torch.device(f"cuda:{find_gpu()}" if torch.cuda.is_available() and find_gpu() is not None else "cpu")
print(f"Using {device}.")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

epochs = 160
batch_size = 128
valid_ratio = None

# 初始化日志和保存目录
save_folder = f'./checkpoint/{args.dataset}/{args.model_type}/'
os.makedirs(save_folder, exist_ok=True)
log_path = os.path.join(save_folder, 'logger.log')
logger = Logger(log_path=log_path)

# 数据准备
print('==> Preparing data..')
trainloader, validloader, testloader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, valid_ratio = args.valid_ratio)

# 模型、损失函数和优化器定义
model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 开始训练
if validloader is not None:
    trained_model = train_model(model, criterion, optimizer, trainloader, validloader, save_folder, epochs=epochs, device = device, logger = logger)
else:
    trained_model = train_model(model, criterion, optimizer, trainloader, testloader, save_folder, epochs=epochs, device = device, logger = logger)

# 测试模型性能
test_model(trained_model, testloader, device, criterion, logger)
