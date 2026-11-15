import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# 数据
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
K = 4

# 训练
km  = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0).fit(X)

# 标签匹配（匈牙利算法），再算准确率
def match(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    m = dict(zip(c, r))
    return np.array([m[p] for p in y_pred])

km_lab, gmm_lab = match(y_true, km.labels_), match(y_true, gmm.predict(X))
print(f"K-Means: Acc={np.mean(km_lab==y_true):.3f}, ARI={adjusted_rand_score(y_true, km.labels_):.3f}")
print(f"GMM:     Acc={np.mean(gmm_lab==y_true):.3f}, ARI={adjusted_rand_score(y_true, gmm.predict(X)):.3f}")

# 画图
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X[:,0], X[:,1], c=y_true,  cmap='viridis', edgecolors='k'); ax[0].set_title('True')
ax[1].scatter(X[:,0], X[:,1], c=km_lab,  cmap='viridis', edgecolors='k'); ax[1].set_title('K-Means')
ax[2].scatter(X[:,0], X[:,1], c=gmm_lab, cmap='viridis', edgecolors='k'); ax[2].set_title('GMM')
plt.tight_layout(); plt.show()