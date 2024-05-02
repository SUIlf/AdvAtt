import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from .utility import SubsetRandomSampler, SubsetSampler, HybridBatchSampler

def cifar10(batch_size, valid_ratio = None, shuffle = True, augmentation = True, train_subset = None):
    """
    Parameters
    ----------
    batch_size : int
        batch size
    valid_ratio : float, optional
        验证集的比例, by default None
    shuffle : bool, optional
        是否随机打乱, by default True
    augmentation : bool, optional
        是否进行数据增强, by default True
    train_subset : int, optional
        是否对训练集降采样，选取采样的数量, by default None

    Returns
    -------
    _type_
        训练集，验证集，测试集以及分类种类
    """
    
    # 定义CIFAR-10数据集的预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) if augmentation == True else transforms.Compose([
        transforms.ToTensor()
        ])
    
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_valid)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

    # 加载训练集

    # 加载测试集
  

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if train_subset is None:
        instance_num = len(trainset)
        indices = list(range(instance_num))
    else:
        indices = np.random.permutation(train_subset)
        instance_num = len(indices)
    print('%d instances are picked from the training set' % instance_num)

    # 加载训练集
    if valid_ratio is not None and valid_ratio > 0.:
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]

        if shuffle == True:
            train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = SubsetSampler(train_idx), SubsetSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 0, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True)

    else:
        if shuffle == True:
            train_sampler = SubsetRandomSampler(indices)
        else:
            train_sampler = SubsetSampler(indices)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = True)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True)

    return train_loader, valid_loader, test_loader, classes