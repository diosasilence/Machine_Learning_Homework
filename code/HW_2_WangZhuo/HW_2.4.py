import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import datasets

# 关掉 cvxopt 的求解输出
solvers.options['show_progress'] = False

# ================= 1. 数据 =================
X, y = datasets.load_iris(return_X_y=True)
X, y = X[y != 2][:, 2:], y[y != 2]
y = np.where(y == 0, -1.0, 1.0)          # 标签取 ±1

n, d = X.shape

# ================= 2. 构造对偶 QP =================
K = X @ X.T
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(n))
G = matrix(-np.eye(n))
h = matrix(np.zeros(n))
A = matrix(y.reshape(1, -1))
b_eq = matrix(np.zeros(1))

# ================= 3. 求解 =================
sol = solvers.qp(P, q, G, h, A, b_eq)
alpha = np.array(sol['x']).flatten()

# ================= 4. 恢复 w, b =================
sv = alpha > 1e-5                         # 支持向量索引
w = (alpha[sv] * y[sv]) @ X[sv]
idx = np.where(sv)[0][0]
b = y[idx] - w @ X[idx]

print("w =", w)
print("b =", b)
print("支持向量个数 =", sv.sum())

# ================= 5. 画图 =================
plt.figure(figsize=(6, 5))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1],
            c='red', edgecolors='k', label='class -1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
            c='blue', edgecolors='k', label='class +1')
plt.scatter(X[sv][:, 0], X[sv][:, 1],
            s=180, facecolors='none', edgecolors='green',
            linewidths=2, label='support vectors')

# 横轴范围
x1 = np.array([X[:, 0].min() - 0.3, X[:, 0].max() + 0.3])

# 三条线：w1*x1 + w2*x2 + b = 0, ±1
for level, style, name in [(0, '-', 'decision boundary'),
                           (1, '--', 'margin'),
                           (-1, '--', 'margin')]:
    x2 = -(w[0] * x1 + b - level) / w[1]
    plt.plot(x1, x2, style, color='k', linewidth=2)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.tight_layout()
plt.show()