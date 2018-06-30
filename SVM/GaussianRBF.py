import numpy as np
import matplotlib.pyplot as plt

#直观理解高斯核函数
#高斯核函数也叫径向基函数---- RBF核 Radial Basis Function Kernal


x = np.arange(-4 ,5 ,1)
#构造一个线性不可分的数据集
y = np.array((x >= -2) & (x <= 2),dtype='int')

# 看看一维的x是如何的
plt.scatter(x[y==0], [0]*len(x[y==0]))
plt.scatter(x[y==1], [0]*len(x[y==1]))
plt.show()

#高斯核函数。简化考虑，gamma先指定为常数未考虑
def gaussian(x, l):
    gamma = 1.0
    return np.exp(-gamma * (x - l)**2)

#定2个地标点landmark。
#但实际上，高斯核，对于每个数据点(样本)都是landmark。将m*n的数据映射成m*m的数据
# 适用于样本数m不多，特征n很多的场景。例如自然语言处理
l1 = -1
l2 = 1

#将一维x，经过高斯核函数计算，得到二维样本X_new
X_new = np.empty((len(x), 2))
#enumerate可以将可遍历的数据对象x(如元组、列表、字符串等)组合为一个索引序列，同时列出数据和数据下标
for i, data in enumerate(x):
    X_new[i, 0] = gaussian(data, l1)
    X_new[i, 1] = gaussian(data, l2)

#看看一维x，经过高斯核函数计算，得到二维样本X_new是如何的
plt.scatter(X_new[y==0,0], X_new[y==0,1])
plt.scatter(X_new[y==1,0], X_new[y==1,1])
plt.show()







