import torch
# 打印PyTorch版本
print(f"PyTorch版本：{torch.__version__}")
# 检查CUDA是否可用
print(f"CUDA是否可用：{torch.cuda.is_available()}")
# 创建测试张量
x = torch.tensor([1, 2, 3])
print(f"测试张量：{x}")
