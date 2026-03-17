import matplotlib
matplotlib.use('TkAgg')  # Windows 下常用后端，也可以试 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. 构造二维数据
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])

# 生成网格点用于绘制分类边界
xx, yy = np.meshgrid(np.linspace(0,10,200),
                    np.linspace(0,10,200))

# 2. 不同K值的分类效果
for k in [1,3,5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(6,4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, s=100, edgecolors='black', cmap=plt.cm.Spectral)
    plt.title(f"KNN Classification (K = {k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.grid()
plt.show()

# 扩展：验证过拟合（K=1）和欠拟合（K过大）
k_list = [1, 2, 3, 4, 5,]
train_acc = []
# 构造测试集
X_test = np.array([[1.5,2.5],[7.5,6.5],[4,4]])
y_test = np.array([0,1,0])

for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    train_acc.append(model.score(X, y))
    test_acc = model.score(X_test, y_test)
    print(f"K={k}时，训练集准确率={model.score(X, y):.2f}，测试集准确率={test_acc:.2f}")

# 绘制准确率曲线
plt.figure()
plt.plot(k_list, train_acc, marker='o', label="Train Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("K值对KNN准确率的影响")
plt.grid()
plt.legend()
plt.show()