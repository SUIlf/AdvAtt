import sys
sys.path.insert(0, './')
import numpy as np

from dataset.cifar10 import cifar10
from dataset.cifar100 import cifar100
# from dataset.svhn import svhn, svhn_plus
# from dataset.cifar100 import cifar100
# from dataset.imagenet100 import imagenet100


# 根据指定的数据集名称加载相应的数据集，并返回训练、验证、测试数据加载器以及类别信息
def parse_data(name, batch_size, valid_ratio = None, shuffle = True, augmentation = True, train_subset = None):

    # 根据数据集名称选择相应的数据加载器
    if name.lower() in ['cifar10',]:
        # CIFAR-10数据集
        trainloader, validloader, testloader, classes = cifar10(batch_size = batch_size, valid_ratio = valid_ratio)
    elif name.lower() in ['cifar100']:
        # CIFAR-100数据集
        trainloader, validloader, testloader, classes = cifar100(batch_size = batch_size, valid_ratio = valid_ratio)
    # elif name.lower() in ['svhn',]:
    #     # SVHN数据集
    #     train_loader, valid_loader, test_loader, classes = svhn(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    # elif name.lower() in ['imagenet100', 'imagenet']:
    #     # ImageNet数据集
    #     train_loader, valid_loader, test_loader, classes = imagenet100(batch_size = batch_size, valid_ratio = valid_ratio, shuffle = shuffle, augmentation = augmentation, **kwargs)
    else:
        raise ValueError('Unrecognized name of the dataset: %s' % name)

    # 返回数据加载器和类别信息
    return trainloader, validloader, testloader, classes
