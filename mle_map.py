import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 1. 构造0-1样本数据并计算MLE和MAP
data = torch.tensor([1.,1.,0.,1.,0.])
p_mle = torch.mean(data)
alpha = 2
beta_param = 2  # 避免与scipy的beta冲突
p_map = (torch.sum(data) + alpha - 1) / (len(data) + alpha + beta_param - 2)

print("MLE =", p_mle.item())
print("MAP =", p_map.item())

# 2. 可视化分析
data_np = np.array([1,1,0,1,0])
N = len(data_np)
sum_x = np.sum(data_np)
p = np.linspace(0,1,100)

# 计算似然、先验、后验
likelihood = p**sum_x * (1-p)**(N - sum_x)
prior = beta.pdf(p, alpha, beta_param)
# 后验正比于似然*先验，这里简化为乘积形式
posterior = likelihood * prior

# 绘图
plt.plot(p, likelihood, label="Likelihood")
plt.plot(p, prior, label="Prior")
plt.plot(p, posterior, label="Posterior")
plt.axvline(p_mle.item(), color='r', linestyle='--', label="MLE")
plt.axvline(p_map.item(), color='g', linestyle='--', label="MAP")
plt.legend()
plt.title("MLE vs MAP")
plt.xlabel("p")
plt.ylabel("Probability Density")
plt.grid()
plt.show()

# 扩展：样本数量增加的影响验证
def compare_mle_map(n_samples):
    """验证样本数量对MLE和MAP差异的影响"""
    np.random.seed(42)
    data_large = np.random.binomial(1, 0.6, n_samples)
    p_mle_large = np.mean(data_large)
    p_map_large = (np.sum(data_large) + alpha -1) / (n_samples + alpha + beta_param -2)
    print(f"\n样本数={n_samples}时：")
    print(f"MLE={p_mle_large:.4f}, MAP={p_map_large:.4f}")
    print(f"差值={abs(p_mle_large - p_map_large):.4f}")

# 测试不同样本量
compare_mle_map(5)
compare_mle_map(100)
compare_mle_map(1000)