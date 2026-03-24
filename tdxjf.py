#梯度下降法
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 替换为你系统有的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
import torch
import numpy as np
import matplotlib.pyplot as plt

# 3.1 基础梯度下降实现
x = torch.tensor([5.0], requires_grad=True)
lr = 0.1
loss_list = []
x_path = [x.item()]

for i in range(20):
    # 定义目标函数 f(x) = x² + 2x + 1
    y = x ** 2 + 2 * x + 1
    loss_list.append(y.item())

    # 反向传播计算梯度
    y.backward()

    # 更新参数（禁止梯度跟踪）
    with torch.no_grad():
        x -= lr * x.grad

    # 清空梯度（避免累积）
    x.grad.zero_()
    x_path.append(x.item())

print(f"最终参数值：x = {x.item():.4f}")
print(f"最终损失值：y = {y.item():.4f}")

# 3.2 损失曲线可视化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_list, marker='o', color='b')
plt.title("Loss Curve (lr=0.1)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()

# 3.3 优化路径可视化
x_vals = np.linspace(-5, 5, 100)
y_vals = x_vals ** 2 + 2 * x_vals + 1

plt.subplot(1, 2, 2)
plt.plot(x_vals, y_vals, color='gray')
plt.scatter(x_path, [xx ** 2 + 2 * xx + 1 for xx in x_path], color='r', s=50)
plt.title("Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.tight_layout()
plt.show()


# 扩展：学习率对收敛的影响
def gradient_descent(lr, iterations=20):
    """不同学习率的梯度下降对比"""
    x = torch.tensor([5.0], requires_grad=True)
    loss_history = []
    for i in range(iterations):
        y = x ** 2 + 2 * x + 1
        loss_history.append(y.item())
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
    return loss_history


# 测试不同学习率
lr_list = [0.01, 0.1, 0.3, 0.9]
plt.figure()
for lr in lr_list:
    loss = gradient_descent(lr)
    plt.plot(loss, label=f"lr={lr}")

plt.title("不同学习率的损失曲线")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()