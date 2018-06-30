import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost1(x):
    return -1 * np.log(x)

def cost0(x):
    return -1 * np.log(1-x)

x = np.linspace(-10, 10 ,500)
x1 = np.linspace(0,1,100)


plt.plot(x, sigmoid(x),color='b',label="sigmoid")
plt.plot(x1, cost1(x1), color='y', label="cost1" )
plt.plot(x1, cost0(x1), color='r', label="cost0" )
plt.legend()
plt.show()