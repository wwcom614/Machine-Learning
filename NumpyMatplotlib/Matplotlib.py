#import matplotlib as mpl

#一般画图用这个就够用了
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(1,10,100)
siny = np.sin(x)
cosy = np.cos(x)

#折线图用plot。使用场景：横轴特征，纵轴标识
plt.plot(x, siny, label="sin(x)", linestyle="--")  #线条样式  : 是纯点   -, 是线点  --是虚线 -是实线
plt.plot(x, cosy, label="cos(x)", color="red") #也可以用RGB
plt.axis([-1,11,-1.5,2])
'''
plt.xlim(-1,11)
plt.ylim(-1.5,2)
'''
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend() #加上label图例
plt.title("Matplotlib plot")
plt.show()

#散点图用scatter。使用场景：横轴特征，纵轴特征
a = np.random.normal(loc=0, scale=1, size=10000)
b = np.random.normal(0,1,10000)
#alpha透明度
plt.scatter(a,b, alpha=0.2)
plt.axis([-4,5,-4.5,5.5])

plt.xlabel("a Axis")
plt.ylabel("b Axis")

plt.title("Matplotlib  scatter")
plt.show()
