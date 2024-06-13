import os
import time
import sys
sys.path.insert(0, './')

import numpy as np
import torch
import torch.nn as nn

from util.data_parser import parse_data
from util.model_utils import model_load
from util.utility import find_gpu, Logger

from util.model_utils import train_model, test_model

# 设备选择
device = torch.device(f"cuda:{find_gpu()}" if torch.cuda.is_available() and find_gpu() is not None else "cpu")
print(f"Using {device}.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = 'cifar100'
batch_size = 128
model_type = 'resnet18'

criterion = nn.CrossEntropyLoss()

# 假设你的数据加载函数可以这样调用（确保使用与训练时相同的参数）
_, _, testloader, _ = parse_data(name=dataset, batch_size=batch_size, valid_ratio=None)

trained_model = model_load(dataset=dataset, model_type=model_type, model_path='./checkpoint', normalize=None)
trained_model = trained_model.to(device)
trained_model.eval()  # 设置为评估模式

test_model(trained_model, testloader, device, criterion)