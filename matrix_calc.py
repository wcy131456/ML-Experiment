import torch
# 定义2x2矩阵A和B
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
# 矩阵乘法（torch.matmul是矩阵乘，*是元素乘）
C = torch.matmul(A, B)
# 打印结果
print("矩阵A：\n", A)
print("矩阵B：\n", B)
print("A×B结果：\n", C)