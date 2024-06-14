import torch
import torch.nn.functional as F
import time

# ----------------------------fgsm_attack-------------------------------------------------
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

# ---------------------------pgd_attack--------------------------------------------------
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

# -----------------------------nuclear_norm------------------------------------------------
# def nuclear_norm(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
#     assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
#     # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
#     if D.ndim == 3:
#         D = D.unsqueeze(0)  # 转换为 (1, C, H, W)
#     # 假设 device 和 radius 已经定义
#     v_FW = torch.zeros_like(D).to(device)
#     # 对每个批次和每个通道应用 SVD
#     for i in range(D.size(0)):  # 遍历批次
#         for j in range(D.size(1)):  # 遍历通道
#             U, _, V = torch.svd(D[i, j])
#             # 向量外积
#             v_FW[i, j] = radius * (U[:, 0:1] @ V[:, 0:1].t())
#     # 如果初始 'D' 是三维的，去除添加的批次维度
#     if D.size(0) == 1 and D.ndim == 3:
#         v_FW = v_FW.squeeze(0)
#     return v_FW

# def nuclear_norm(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
#     assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
#     # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
#     if D.ndim == 3:
#         D = D.unsqueeze(0)  # 转换为 (1, C, H, W)
#     # 假设 device 和 radius 已经定义
#     ori_shape = D.size()
#     U, s, V = torch.linalg.svd(D.view(ori_shape[0], ori_shape[1]*ori_shape[2], -1), full_matrices=False)
#     prj_pt = radius * torch.einsum('bi,bj->bij', U[:,:,0], V[:,0,:])
#     v_FW = prj_pt.view(ori_shape)
    
#     return v_FW

# ---------------------------------spectral_norm--------------------------------------------
def norms(D: torch.Tensor, radius: float, norm_type='nuclear', device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    if norm_type == 'nuclear':
        k = 1
    elif norm_type == 'spectral':
        k = max(1, int(D.size(2) * 0.5))  # 假设 k 是通道数的 50%
    # 创建一个尺寸为 (k, k) 的对角矩阵，对角线元素为 radius/k
    diag_matrix = torch.diag(torch.full((k,), radius/k, dtype=torch.float)).to(device)
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

def FW_nuclear_attack(model, images, labels, radius=1, eps=1e-10, step_size=0.1, iters=20, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad = True
    for iteration in range(iters):
        perturbation.requires_grad_()
        with torch.enable_grad():
            outputs = model(torch.clamp(images + perturbation, 0, 1))
            loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, [perturbation], retain_graph=True)[0]
        # Compute optimal perturbation
        v_FW = norms(-grad, radius, norm_type = 'nuclear', device=device)
        # step_size = step_size * 1. / (iteration + 1)
        step_size = 2/(iteration + 2)
        perturbation = perturbation + step_size * (v_FW - perturbation)
        adv_images = images.detach() + perturbation.detach()
        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break
    adv_images = images.detach() + perturbation.detach()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images


def FW_spectral_attack(model, images, labels, radius=1, eps=1e-10, step_size=0.1, iters=20, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad = True
    for iteration in range(iters):
        perturbation.requires_grad_()
        with torch.enable_grad():
            outputs = model(torch.clamp(images + perturbation, 0, 1))
            loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, [perturbation], retain_graph=True)[0]
        # Compute optimal perturbation
        v_FW = norms(-grad, radius, norm_type = 'spectral', device=device)
        # step_size = step_size * 1. / (iteration + 1)
        step_size = 2/(iteration + 2)
        perturbation = perturbation + step_size * (v_FW - perturbation)
        adv_images = images.detach() + perturbation.detach()
        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break
    adv_images = images.detach() + perturbation.detach()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images



# def FW_spectral_attack(model, images, labels, radius=1, eps=1e-10, step_size=1.0, iters=40, device='cpu'):
#     criterion = torch.nn.CrossEntropyLoss()
#     perturbation = torch.zeros_like(images).to(device)
#     perturbation.requires_grad_()
#     for iteration in range(iters):
#         outputs = model(images + perturbation)
#         loss = -criterion(outputs, labels)
#         grad = torch.autograd.grad(loss, perturbation)[0]

#         # Compute optimal perturbation
#         v_FW = norms(-grad, radius, norm_type = 'spectral', device=device)
#         step_size = step_size * 1.0 / (iteration + 3)
#         perturbation.data += step_size * (v_FW - perturbation)

#         # Early stopping if the gradient updates become very small
#         if torch.norm(grad) < eps:
#             break

#     adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
#     return adv_images

# ---------------------------------linf_norm--------------------------------------------
def linf_norm(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    k = max(1, int(D.size(2) * 0.5))  # 假设 k 是通道数的 50%
    # 创建一个尺寸为 (k, k) 的对角矩阵，对角线元素为 radius
    diag_matrix = torch.diag(torch.full((k,), radius, dtype=torch.float)).to(device)
    # 创建一个与 D 形状相同的张量来存储结果，初始化为 0 并移至相应设备
    v_FW = torch.zeros_like(D).to(device)
    
    # Compute optimal perturbation using L_inf norm
    norm = torch.norm(D.view(D.shape[0], -1), float('inf'), dim=1).view(-1, 1, 1, 1)
    norm = torch.max(norm, torch.tensor([1e-10]).to(device))  # Avoid division by zero
    v_FW = radius * torch.sign(D) * (norm <= radius) + D * (norm > radius)
    
    
    # 如果原始 'D' 是三维的，去除添加的批次维度
    if D.size(0) == 1 and D.ndim == 3:
        v_FW = v_FW.squeeze(0)
    return v_FW

def FW_Linf_attack(model, images, labels, radius=1, eps=1e-10, step_size=1.0, iters=40, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad_()
    for iteration in range(iters):
        outputs = model(images + perturbation)
        loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, perturbation)[0]

        # Compute optimal perturbation
        v_FW = linf_norm(-grad, radius, device=device)
        step_size = step_size * 1.0 / (iteration + 1)
        perturbation.data += step_size * (v_FW - perturbation)

        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break

    adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
    return adv_images

# ---------------------------------l2_norm--------------------------------------------
def l2_norm(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    k = max(1, int(D.size(2) * 0.5))  # 假设 k 是通道数的 50%
    # 创建一个尺寸为 (k, k) 的对角矩阵，对角线元素为 radius
    diag_matrix = torch.diag(torch.full((k,), radius, dtype=torch.float)).to(device)
    # 创建一个与 D 形状相同的张量来存储结果，初始化为 0 并移至相应设备
    v_FW = torch.zeros_like(D).to(device)
    
    norm = torch.norm(D.view(D.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
    norm = torch.max(norm, torch.tensor([1e-10]).to(device))  # Avoid division by zero
    v_FW = (radius / norm) * D
    
    
    # 如果原始 'D' 是三维的，去除添加的批次维度
    if D.size(0) == 1 and D.ndim == 3:
        v_FW = v_FW.squeeze(0)
    return v_FW

def FW_L2_attack(model, images, labels, radius=1, eps=1e-10, step_size=1.0, iters=40, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad_()
    for iteration in range(iters):
        outputs = model(images + perturbation)
        loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, perturbation)[0]

        # Compute optimal perturbation
        v_FW = l2_norm(-grad, radius, device=device)
        step_size = step_size * 1.0 / (iteration + 1)
        perturbation.data += step_size * (v_FW - perturbation)

        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break

    adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
    return adv_images

# ---------------------------------l1_norm--------------------------------------------
def l1_norm(D: torch.Tensor, radius: float, device='cpu') -> torch.Tensor:
    assert isinstance(D, torch.Tensor), "Input must be a PyTorch tensor."
    # 假设 'D' 已经是一个至少三维的张量，形状可能是 (C, H, W) 或 (B, C, H, W)
    if D.ndim == 3:
        D = D.unsqueeze(0)  # 将其转换为 (1, C, H, W)
    k = max(1, int(D.size(2) * 0.5))  # 假设 k 是通道数的 50%
    # 创建一个尺寸为 (k, k) 的对角矩阵，对角线元素为 radius
    diag_matrix = torch.diag(torch.full((k,), radius, dtype=torch.float)).to(device)
    # 创建一个与 D 形状相同的张量来存储结果，初始化为 0 并移至相应设备
    v_FW = torch.zeros_like(D).to(device)

    norm = torch.norm(D.view(D.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
    norm = torch.max(norm, torch.tensor([1e-10]).to(device))  # Avoid division by zero
    v_FW = (radius / norm) * D
    
    
    # 如果原始 'D' 是三维的，去除添加的批次维度
    if D.size(0) == 1 and D.ndim == 3:
        v_FW = v_FW.squeeze(0)
    return v_FW

def FW_L1_attack(model, images, labels, radius=1, eps=1e-10, step_size=1.0, iters=40, device='cpu'):
    criterion = torch.nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(images).to(device)
    perturbation.requires_grad_()
    for iteration in range(iters):
        outputs = model(images + perturbation)
        loss = -criterion(outputs, labels)
        grad = torch.autograd.grad(loss, perturbation)[0]

        # Compute optimal perturbation
        v_FW = l1_norm(-grad, radius, device=device)
        step_size = step_size * 1.0 / (iteration + 1)
        perturbation.data += step_size * (v_FW - perturbation)

        # Early stopping if the gradient updates become very small
        if torch.norm(grad) < eps:
            break

    adv_images = torch.clamp(images + perturbation.detach(), 0, 1)
    return adv_images