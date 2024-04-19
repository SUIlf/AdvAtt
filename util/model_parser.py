import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import pdb

from models.preprocess import DataNormalizeLayer
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101


# mnist_normalize = {'bias': [0.5, 0.5, 0.5], 'scale': [0.5, 0.5, 0.5]}
# svhn_normalize = {'bias': [0.4380, 0.4440, 0.4730], 'scale': [0.1751, 0.1771, 0.1744]}
# cifar10_normalize = {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]}
# cifar100_normalize = {'bias': [0.5071, 0.4867, 0.4408], 'scale': [0.2675, 0.2565, 0.2761]}

normalize_dict = {
    'mnist': {'bias': [0.5, 0.5, 0.5], 'scale': [0.5, 0.5, 0.5]},
    'cifar10': {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'bias': [0.5071, 0.4867, 0.4408], 'scale': [0.2675, 0.2565, 0.2761]},
    'imagenet100': {'bias': [0.485, 0.456, 0.406], 'scale': [0.229, 0.224, 0.225]},
}

num_classes_dict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet100': 100,
}

def parse_model(dataset, model_type, normalize = None, **kwargs):
    '''
    根据指定的数据集和模型类型创建神经网络模型
    '''
    # 检查数据集是否在支持的列表中
    assert dataset in ['cifar10', 'cifar100', 'mnist', 'imagenet100'], 'Dataset not included!'

    # 配置标准化层
    if normalize is not None:
        # 如果提供了标准化参数
        normalize_layer = DataNormalizeLayer(bias = normalize_dict[dataset]['bias'], scale = normalize_dict[dataset]['scale'])
    else:
        # 如果未提供标准化参数，默认不进行标准化
        normalize_layer = DataNormalizeLayer(bias = 0., scale = 1.)

    # 获取数据集对应的类别数
    num_classes = num_classes_dict[dataset]

    # 创建相应的神经网络模型
    if model_type.lower() in ['resnet', 'resnet18']:
        net = ResNet18(num_classes=num_classes)
    elif model_type.lower() in ['resnet34']:
        net = ResNet34(num_classes=num_classes)
    elif model_type.lower() in ['resnet50']:
        net = ResNet50(num_classes=num_classes)
    elif model_type.lower() in ['resnet101']:
        net = ResNet101(num_classes=num_classes)
    else:
        raise ValueError('Unrecognized architecture: %s' % model_type)

    # 将标准化层和模型组合为一个序列
    return nn.Sequential(normalize_layer, net)

