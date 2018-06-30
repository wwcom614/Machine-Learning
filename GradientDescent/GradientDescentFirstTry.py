import numpy as np
import matplotlib.pyplot as plt

# 梯度下降法求 y = J(x)= (theta - 2.5) ** 2 -1 的极小值
def J(theta):
    try:
        return (theta - 2.5) ** 2 -1
    except:
        return float('inf')

def dJ(theta):
    return 2 * (theta - 2.5)

x = np.linspace(-1, 6, 141)
y = J(x)

eta = 0.01 #学习率，梯度下降步长，一般经验取0.01
epsilon = 1e-8 #结束精度，两次移动的损失函数差值低于该值停止梯度下降循环
n_iters = 1e4 #梯度下降算法最大循环次数(避免死循环)
n = 1 #当前循环次数
theta = 0.0 # 横坐标
theta_history = [theta] # 横坐标theta的移动轨迹
while n < n_iters:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient #横坐标theta每次移动 步长*导数 的距离
    theta_history.append(theta)
    if(abs(J(theta) - J(last_theta)) < epsilon):
        break
    n += 1

print("theta:", theta)
#theta: 2.4995140741236224
print("J(theta)", J(theta))
#J(theta) -0.9999997638760426
print("总共走的步数：",len(theta_history))
#总共走的步数： 424
plt.plot(x, y)
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r',marker='+')
plt.show()



