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

def LP_nuclear_gpu(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    # 假设 device 和 radius 已经定义
    v_FW = torch.zeros_like(D).to(device)
    # 对每个批次和每个通道应用 SVD
    for i in range(D.size(0)):  # 遍历批次
        for j in range(D.size(1)):  # 遍历通道
            U, _, V = torch.svd(D[i, j])
            # 使用广播机制处理向量外积
            v_FW[i, j] = radius * (U[:, 0:1] @ V[:, 0:1].t())
    # 如果原始 'D' 是三维的，去除添加的批次维度
    if D.size(0) == 1 and D.ndim == 3:
        v_FW = v_FW.squeeze(0)
    return v_FW

def FW_nuclear_attack(model, images, labels, radius=1, eps=1e-10, step_size=0.1, T_max=40, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad_()
    for iteration in range(T_max):
        outputs = model(images + perturbation)
        loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, perturbation)[0]
        # Compute optimal perturbation
        v_FW = LP_nuclear_gpu(-grad, radius, device=device)
        step_size = step_size * 1.0 / (iteration + 1)
        perturbation.data += step_size * (v_FW - perturbation)
        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break
    adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
    return adv_images

def LP_spectral_gpu(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    k = max(1, int(D.size(2) * 1))  # 假设 k 是通道数的 60%
    # 创建一个尺寸为 (k, k) 的对角矩阵，对角线元素为 radius
    diag_matrix = torch.diag(torch.full((k,), radius, dtype=torch.float)).to(device)
    # 创建一个与 D 形状相同的张量来存储结果，初始化为 0 并移至相应设备
    v_FW = torch.zeros_like(D).to(device)
    # 对每个批次和每个通道应用 SVD
    for i in range(D.size(0)):  # 遍历批次
        for j in range(D.size(1)):  # 遍历通道
            U, _, V = torch.svd(D[i, j])
            # 仅使用前 k 个奇异向量，确保 U 和 V 至少有 k 列
            U_k = U[:, :k]
            V_k = V[:, :k]
            # 计算外积，并使用对角矩阵进行缩放
            v_FW[i, j] = (U_k @ diag_matrix @ V_k.t())
    # 如果原始 'D' 是三维的，去除添加的批次维度
    if D.size(0) == 1 and D.ndim == 3:
        v_FW = v_FW.squeeze(0)
    return v_FW

def FW_spectral_attack(model, images, labels, radius=1, eps=1e-10, step_size=1.0, T_max=40, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad_()
    for iteration in range(T_max):
        outputs = model(images + perturbation)
        loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, perturbation)[0]

        # Compute optimal perturbation
        v_FW = LP_spectral_gpu(-grad, radius, device=device)
        step_size = step_size * 1.0 / (iteration + 1)
        perturbation.data += step_size * (v_FW - perturbation)

        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break

    adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
    return adv_images