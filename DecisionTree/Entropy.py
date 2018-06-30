import numpy as np
import matplotlib.pyplot as plt

#绘制二元信息熵
#熵越大，数据的不确定性越高
#熵越小，数据的不确定性越低
#等概率出现熵最大，因为此时数据最不确定
def entropy(p):
    return -p * np.log(p) - (1-p) * np.log(1-p)

x = np.linspace(0.01, 0.99, 200)
plt.plot(x, entropy(x))
plt.show()