import os
import sys
sys.path.insert(0, './')

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from util.data_parser import parse_data
from util.model_utils import model_load
from util.utility import find_gpu, Logger
from util.attacks import fgsm_attack, pgd_attack, FW_nuclear_attack, FW_spectral_attack

def imshow(img_T, ax, title=None):
    """显示单个图像的函数，进行反标准化并调整维度以适配 matplotlib 显示"""
    img = img_T[0].numpy()  # 反标准化
    img = np.clip(img, 0, 1)  # 确保图像数据在[0, 1]范围内
    ax.imshow(np.transpose(img, (1, 2, 0)))
    if title:
        ax.set_title(title, fontsize=30)
    ax.axis('off')


# 初始化
dataset = 'cifar10'
batch_size = 5
model_type = 'resnet18'
device = torch.device(f"cuda:{find_gpu()}" if torch.cuda.is_available() and find_gpu() is not None else "cpu")
print(f"Using {device}.")

save_folder = f'./checkpoint1/{dataset}/{model_type}/'
os.makedirs(save_folder, exist_ok=True)
logger = Logger(os.path.join(save_folder, 'logger.log'))

_, _, testloader, _ = parse_data(name=dataset, batch_size=batch_size, valid_ratio=None)

model = model_load(dataset=dataset, model_type=model_type, model_path='./checkpoint', normalize=None).to(device).eval()

epsilons = [0, 0.05, 0.1, 0.5, 1, 5, 10]
# attack_types = ['fgsm', 'pgd', 'l2', 'linf', 'nuclear']
attack_types = ['nuclear', 'spectral']
alpha_pgd = 1
iters_pgd = 10

# 测试攻击
for attack_type in attack_types:
    fig, axes = plt.subplots(6, len(epsilons), figsize=(40, 40))
    for i, (images, labels) in enumerate(testloader):
        if i >= 3: break
        images, labels = images.to(device), labels.to(device)
        original_data_np = images.detach().cpu()

        for j, epsilon in enumerate(epsilons):
            # 应用攻击
            if attack_type == 'fgsm':
                perturbed_images = fgsm_attack(model, images, labels, epsilon)
            elif attack_type == 'pgd':
                perturbed_images = pgd_attack(model, images, labels, eps=epsilon, alpha=alpha_pgd, iters=iters_pgd)
            # elif attack_type == 'l2':
            #     perturbed_images = L2_norm_attack(model, images, labels, eps=epsilon, alpha=alpha_pgd, iters=iters_pgd, device=device)
            # elif attack_type == 'linf':
            #     perturbed_images = Linf_norm_attack(model, images, labels, eps=epsilon, alpha=alpha_pgd, iters=iters_pgd, device=device)
            # elif attack_type == 'nuclear':
            #     perturbed_images = nuclear_norm_attack(model, images, labels, eps=epsilon, alpha=alpha_pgd, iters=iters_pgd, device=device)
            elif attack_type == 'nuclear':
                perturbed_images = FW_nuclear_attack(model, images, labels, radius=epsilon, eps=1e-10, step_size=1.0, T_max=40, device=device)
            elif attack_type == 'spectral':
                perturbed_images = FW_spectral_attack(model, images, labels, radius=epsilon, eps=1e-10, step_size=1.0, T_max=40, device=device)
            
            perturbed_data_np = perturbed_images.detach().cpu()

            ax = axes[2*i, j]
            ax_perturbation = axes[i*2+1, j]
            imshow(perturbed_data_np, ax, f"Eps: {epsilon}")
            imshow(perturbed_data_np - original_data_np, ax_perturbation, f"Delta: {epsilon}")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{attack_type}-perturbation.png")
    print(attack_type, 'completed')
