import torch
import torch.nn.functional as F
import time

def fgsm_attack(model, images, labels, epsilon):
    # 设置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 激活图像的梯度属性
    images.requires_grad = True
    output = model(images)
    model.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    image_grad = images.grad.data    
    perturbed_images = images + epsilon * image_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

def pgd_attack(model, images, labels, eps, alpha, iters=40):
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    original_images = images.data.clone()
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        perturbed_images = images + alpha * images.grad.sign()        
        eta = torch.clamp(perturbed_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, 0, 1).detach_()
    return images


# test