import torch

# 假设的张量形状和设备
U = torch.randn(32, 32, device='cuda:9')
V = torch.randn(32, 32, device='cuda:9')
v_FW = torch.zeros((1, 1, 32, 32), device='cuda:9')  # 假设的目标张量

# 尝试复现操作
try:
    v_FW[0, 0] = 1.5 * (U[:, 0:1] @ V[:, 0:1].t())  # 简化的类似操作
    print("Operation successful, result:", v_FW[0, 0])
except Exception as e:
    print("Error occurred:", e)
