# ==============================================
# 智慧农业温室时序数据的番茄产量预测 - 完整实验代码
# 路径：D:\ML-Experiment
# 环境：虚拟环境
# ==============================================
import matplotlib.pyplot as plt

# 替换为系统支持的中文字体（Windows 通用）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 调整图表布局，避免标签被截断
plt.rcParams['figure.autolayout'] = True
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===================== 0. 固定路径 =====================
BASE_PATH = r"D:\ML-Experiment"
DATA_FILE = r"D:\ML-Experiment\smart_agri_tomato_timeseries_raw\smart_agri_tomato_timeseries_raw.xls"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取数据 =====================
print("=" * 50)
print("正在读取数据...")
df = pd.read_csv(DATA_FILE, encoding='utf-8')
print(f"数据集形状：{df.shape}")
print("数据读取完成！")

# ===================== 2. 缺失值检查 =====================
print("\n" + "=" * 50)
print("缺失值统计：")
missing = df.isnull().sum()
print(missing[missing > 0])

# ===================== 3. EDA 可视化 =====================
print("\n开始绘制EDA图表...")

# 缺失值条形图
plt.figure(figsize=(10, 5))
missing.plot(kind='bar', color='#ff6b6b')
plt.title('各字段缺失值统计')
plt.tight_layout()
plt.show()

# 选择数值列
numeric_cols = [
    "temp", "humidity", "light", "co2", "irrigation",
    "ph", "canopy_temp", "temp_24h_mean", "light_24h_sum",
    "co2_24h_mean", "growth_index", "yield_now", "yield_next_24h"
]
available_numeric = [c for c in numeric_cols if c in df.columns]

# 分布直方图
df[available_numeric].hist(figsize=(16, 12), bins=20, color='#4ecdc4')
plt.suptitle('特征分布直方图')
plt.tight_layout()
plt.show()

# 相关性热力图
corr = df[available_numeric].corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.columns)
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.show()

# ===================== 4. 数据预处理 Pipeline =====================
print("\n" + "=" * 50)
print("构建预处理流水线...")

# 目标变量
target = "yield_next_24h"
X = df.drop([target, "timestamp"], axis=1, errors='ignore')
y = df[target]

# 特征分类
numeric_features = [c for c in [
    "temp", "humidity", "light", "co2", "irrigation", "fertilizer_ec",
    "ph", "canopy_temp", "temp_24h_mean", "light_24h_sum",
    "co2_24h_mean", "irrigation_24h_sum", "growth_index", "yield_now"
] if c in X.columns]

categorical_features = [c for c in ["greenhouse_id"] if c in X.columns]

# 预处理
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ===================== 5. 时序数据切分（防泄漏） =====================
print("按时间顺序划分训练集/测试集...")
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
X = df_sorted.drop([target, "timestamp"], axis=1, errors='ignore')
y = df_sorted[target]

split_idx = int(len(df_sorted) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"训练集：{X_train_processed.shape}")
print(f"测试集：{X_test_processed.shape}")

# ===================== 6. 手写梯度下降线性回归 =====================
print("\n" + "=" * 50)
print("训练 手写梯度下降 模型...")

X_train_gd = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_gd = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

# 加偏置项
X_train_b = np.c_[np.ones((X_train_gd.shape[0], 1)), X_train_gd]
X_test_b = np.c_[np.ones((X_test_gd.shape[0], 1)), X_test_gd]
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 梯度下降
theta = np.zeros((X_train_b.shape[1], 1))
lr = 0.01
epochs = 500
loss_history = []

for epoch in range(epochs):
    y_pred = X_train_b @ theta
    error = y_pred - y_train_np
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    grad = (2 / len(X_train_b)) * X_train_b.T @ error
    theta = theta - lr * grad

# 测试
y_pred_gd = X_test_b @ theta
mse_gd = mean_squared_error(y_test_np, y_pred_gd)
r2_gd = r2_score(y_test_np, y_pred_gd)

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='#2ecc71')
plt.title("手写GD损失曲线")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.show()

# ===================== 7. Sklearn 线性回归 =====================
print("\n训练 Sklearn 线性回归模型...")
model_sk = Pipeline([
    ("pre", preprocessor),
    ("lr", LinearRegression())
])
model_sk.fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)
mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)

# ===================== 8. PyTorch 线性回归 =====================
print("\n训练 PyTorch 模型...")
X_train_t = torch.tensor(X_train_gd, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
X_test_t = torch.tensor(X_test_gd, dtype=torch.float32)
y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(train_ds, batch_size=32, shuffle=False)

class TorchModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.fc(x)

model_t = TorchModel(X_train_gd.shape[1])
criterion = nn.MSELoss()
opt = optim.SGD(model_t.parameters(), lr=0.01)

loss_t_history = []
model_t.train()
for epoch in range(500):
    total = 0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model_t(xb)
        loss = criterion(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    loss_t_history.append(total / len(train_ds))

# 评估
model_t.eval()
with torch.no_grad():
    y_pred_t = model_t(X_test_t)
mse_t = criterion(y_pred_t, y_test_t).item()
r2_t = r2_score(y_test_np, y_pred_t.numpy())

# ===================== 9. 结果对比 =====================
print("\n" + "=" * 50)
print("            模型结果对比")
print("=" * 50)
print(f"手写GD    | MSE={mse_gd:.4f}  R2={r2_gd:.4f}")
print(f"Sklearn   | MSE={mse_sk:.4f}  R2={r2_sk:.4f}")
print(f"PyTorch   | MSE={mse_t:.4f}  R2={r2_t:.4f}")
print("=" * 50)

# ===================== 10. 残差分析 =====================
print("\n生成残差分析图...")
residuals = y_test - y_pred_sk

plt.figure(figsize=(8, 4))
plt.scatter(y_pred_sk, residuals, alpha=0.6, color='#9b59b6')
plt.axhline(0, color='red', linestyle='--')
plt.title("残差分布")
plt.xlabel("预测值")
plt.ylabel("残差")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred_sk, alpha=0.6, color='#f39c12')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("真实值 vs 预测值")
plt.xlabel("真实产量")
plt.ylabel("预测产量")
plt.grid(alpha=0.3)
plt.show()

print("\n✅ 实验全部完成！")