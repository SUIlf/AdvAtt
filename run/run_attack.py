import os
import argparse
import time
import sys
sys.path.insert(0, './')
import torch
import argparse

from util.data_parser import parse_data
from util.model_utils import model_load
from util.utility import find_gpu, Logger, progress_bar

from util.attacks import fgsm_attack, pgd_attack, L2_norm_attack, Linf_norm_attack, nuclear_norm_attack

# 使用argparse处理命令行参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# dataset，默认值为'cifar10'
parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'The dataset used, default = "cifar10".')
# batch_size，用于指定批量大小
parser.add_argument('--batch_size', type = int, default = 128, help = 'The batch size, default is 128.')
# model_type，用于指定模型类型
parser.add_argument('--model_type', type = str, default = 'resnet18', help = 'The type of the model, default is "resnet18".')
args = parser.parse_args()

# 设备选择
device = torch.device(f"cuda:{find_gpu()}" if torch.cuda.is_available() and find_gpu() is not None else "cpu")
print(f"Using {device}.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化日志和保存目录
save_folder = f'./checkpoint1/{args.dataset}/{args.model_type}/'
os.makedirs(save_folder, exist_ok=True)
log_path = os.path.join(save_folder, 'logger.log')
logger = Logger(log_path=log_path)

# 假设你的数据加载函数可以这样调用（确保使用与训练时相同的参数）
_, _, testloader, _ = parse_data(name=args.dataset, batch_size=args.batch_size, valid_ratio=None)

model = model_load(dataset=args.dataset, model_type=args.model_type, model_path='./checkpoint', normalize=None)
model = model.to(device)
model.eval()  # 设置为评估模式


# 定义 FGSM 和 PGD 攻击的参数
epsilon_fgsm = 0.03  # FGSM 的攻击扰动大小

epsilon_pgd = 8/255  # PGD 的攻击扰动大小
epsilon_l2 = 2/255
epsilon_linf = 0.03
alpha_pgd = 0.1  # PGD 的步长
iters_pgd = 20 # PGD 的迭代次数

# 测试函数
def test(model, device, test_loader, attack_type, epsilon):
    correct = 0
    total = 0
    start_time = time.time()  # 攻击开始时间
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 应用攻击
        if attack_type == 'fgsm':
            perturbed_images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == 'pgd':
            perturbed_images = pgd_attack(model, images, labels, eps=epsilon_pgd, alpha=alpha_pgd, iters=iters_pgd)
        elif attack_type == 'l2':
            perturbed_images = L2_norm_attack(model, images, labels, eps=epsilon, alpha=alpha_pgd, iters=iters_pgd, device=device)
        elif attack_type == 'linf':
            perturbed_images = Linf_norm_attack(model, images, labels, eps=epsilon_pgd, alpha=alpha_pgd, iters=iters_pgd, device=device)
        elif attack_type == 'nuclear':
            perturbed_images = nuclear_norm_attack(model, images, labels, eps=epsilon_pgd, alpha=alpha_pgd, iters=iters_pgd, device=device)

        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar(batch_idx, len(test_loader), f'| Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    end_time = time.time()  # 攻击结束时间
    attack_time = end_time - start_time  # 计算总耗时
    
    accuracy = 100 * correct / total
    logger.log(f'Accuracy of the model under {attack_type.upper()} attack: {accuracy:.2f}%')
    logger.log(f'Total time taken for {attack_type.upper()} attack: {attack_time:.2f} seconds')
    logger.log(f' ')


# 运行测试
print('fgsm')
test(model, device, testloader, 'fgsm', epsilon_pgd)

# print('l2')
# test(model, device, testloader, 'l2', epsilon_pgd)

# print('nuclear')
# test(model, device, testloader, 'nuclear', epsilon_pgd)

# print('pgd')
# test(model, device, testloader, 'pgd', epsilon_pgd)

print('linf')
test(model, device, testloader, 'linf', epsilon_pgd)


