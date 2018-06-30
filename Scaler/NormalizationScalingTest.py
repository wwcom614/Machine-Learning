import numpy as np
import matplotlib.pyplot as plt

X = np.random.randint(0, 100, (20, 2))
X = np.array(X, dtype=float)
print(X)

#最值归一化
X[:,0] = (X[:,0] - np.min(X[:,0])) /(np.max(X[:,0]) - np.min(X[:,0]))
X[:,1] = (X[:,1] - np.min(X[:,1])) /(np.max(X[:,1]) - np.min(X[:,1]))

print(X)
print("np.mean(X[:,0])",np.mean(X[:,0]))
print("np.std(X[:,0])",np.std(X[:,0]))

print("np.mean(X[:,1])",np.mean(X[:,1]))
print("np.std(X[:,1])",np.std(X[:,1]))


plt.scatter(X[:,0], X[:,1])
plt.show()

#均值方差归一化