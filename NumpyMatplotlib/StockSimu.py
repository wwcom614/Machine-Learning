# 模拟绘制股票图
# 相比普通折线图区别：
# 1. y轴上，每个柱子起点不是0，而是beginPrice。
# 2. 股票下跌场景时，柱子有可能beginPrice在上方，endPrice在下方，此时使用绿色柱子；反之正常使用红色柱子
import numpy as np
import matplotlib.pyplot as plt

date = np.linspace(start=1, stop=15, num=15)
beginPrice = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2877.40])
endPrice = np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2919.08])

plt.figure()
for i in range(0, 15):
    dateOne = np.zeros(shape=[2])
    dateOne[0] = i
    dateOne[1] = i
    priceOne = np.zeros(shape=[2])
    priceOne[0] = beginPrice[i]
    priceOne[1] = endPrice[i]
    if endPrice[i] > beginPrice[i]:
        plt.plot(dateOne, priceOne, 'r', lw=8)
        print(dateOne,priceOne)
    else:
        plt.plot(dateOne, priceOne, 'g', lw=8)
        print(dateOne,priceOne)
plt.show()







