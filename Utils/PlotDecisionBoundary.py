import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, axis):

    X0,X1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    #形成一个新的待预测矩阵，这个矩阵是x和y坐标栅格化的点组成的
    X_new = np.c_[X0.ravel(),X1.ravel()]
    y_predict = model.predict(X_new)
    Y_predict = y_predict.reshape(X0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    #在X0, X1平面上，绘制等高线Y_predict，
    # contour f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）
    plt.contourf(X0, X1, Y_predict,cmap=custom_cmap)

# SVM在决策边界上下补充2条支撑向量显示
def plot_svc_decision_boundary(model, axis):

    X0,X1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    #形成一个新的待预测矩阵，这个矩阵是x和y坐标栅格化的点组成的
    X_new = np.c_[X0.ravel(),X1.ravel()]
    y_predict = model.predict(X_new)
    Y_predict = y_predict.reshape(X0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    #在X0, X1平面上，绘制等高线Y_predict，
    # contour f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）
    plt.contourf(X0, X1, Y_predict,linewidth=5,cmap=custom_cmap)

    w = model.coef_[0]
    b = model.intercept_[0]
    # w0 * X0 + w1 * X1 + b = 0
    # => 决策边界： X1 = -w0/w1 * X0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]
    #画图y轴截取
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color = 'black')
    plt.plot(plot_x[down_index], down_y[down_index], color = 'black')

#########################################################
#方法学习

# meshgrid函数用两个坐标轴上的点在平面上画网格
# [X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y,
# 其中矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制
# x是1行m列向量，将向量x复制n行生成矩阵X
# y是1行n列向量，将向量y转置，复制m列生成矩阵X
# 生成的X和Y都是n*m的矩阵
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 3)
X, Y = np.meshgrid(x,y)
#print("X:",X)
'''
X: [[0.   0.25 0.5  0.75 1.  ]
 [0.   0.25 0.5  0.75 1.  ]
 [0.   0.25 0.5  0.75 1.  ]]
'''

#print("Y:",Y)
'''
Y: [[0.  0.  0.  0.  0. ]
 [0.5 0.5 0.5 0.5 0.5]
 [1.  1.  1.  1.  1. ]]
'''

#np.r_是按行连接两个矩阵，要求两个原矩阵行数相等，类似于pandas中的concat()。
#np.c_是按列连接两个矩阵，要求两个原矩阵列数相等，类似于pandas中的merge()。
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.c_[a,b]

#print("np.r_[a,b]:",np.r_[a,b])
#np.r_[a,b]: [1 2 3 4 5 6]

#print("np.c_[a,b]:",np.c_[a,b])
'''
np.c_[a,b]: [[1 4]
 [2 5]
 [3 6]]
'''





#numpy的ravel()和flatten()函数实现的功能是一致的（将多维数组降位一维）。
# ravel(散开，解开)，flatten（变平）
#两者的区别在于返回拷贝（copy）还是返回视图（view）。
# numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
# numpy.ravel()返回的是视图（view，引用reference的意味），会影响（reflects）原始矩阵。

axis = [-1,2,3,5]
x0,x1 = np.meshgrid(
    np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*2)).reshape(-1,1),
    np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*3)).reshape(-1,1)
)
#print("x0:",x0)
'''
x0: [[-1.  -0.4  0.2  0.8  1.4  2. ]
 [-1.  -0.4  0.2  0.8  1.4  2. ]
 [-1.  -0.4  0.2  0.8  1.4  2. ]
 [-1.  -0.4  0.2  0.8  1.4  2. ]
 [-1.  -0.4  0.2  0.8  1.4  2. ]
 [-1.  -0.4  0.2  0.8  1.4  2. ]]
'''

#print("x1:",x1)
'''
x1: [[3.  3.  3.  3.  3.  3. ]
 [3.4 3.4 3.4 3.4 3.4 3.4]
 [3.8 3.8 3.8 3.8 3.8 3.8]
 [4.2 4.2 4.2 4.2 4.2 4.2]
 [4.6 4.6 4.6 4.6 4.6 4.6]
 [5.  5.  5.  5.  5.  5. ]]
'''


x_new = np.c_[x0.ravel(),x1.ravel()]

#print("x0.ravel():",x0.ravel())
'''
x0.ravel(): [-1.  -0.4  0.2  0.8  1.4  2.  -1.  -0.4  0.2  0.8  1.4  2.  -1.  -0.4
0.2  0.8  1.4  2.  -1.  -0.4  0.2  0.8  1.4  2.  -1.  -0.4  0.2  0.8
1.4  2.  -1.  -0.4  0.2  0.8  1.4  2. ]
'''

#print("x1.ravel():",x1.ravel())
'''
x1.ravel(): [3.  3.  3.  3.  3.  3.  3.4 3.4 3.4 3.4 3.4 3.4 3.8 3.8 3.8 3.8 3.8 3.8
 4.2 4.2 4.2 4.2 4.2 4.2 4.6 4.6 4.6 4.6 4.6 4.6 5.  5.  5.  5.  5.  5. ]
'''


#print("x_new:",x_new)
'''
x_new: [[-1.   3. ]
 [-0.4  3. ]
 [ 0.2  3. ]
 [ 0.8  3. ]
 [ 1.4  3. ]
 [ 2.   3. ]
 [-1.   3.4]
 [-0.4  3.4]
 [ 0.2  3.4]
 [ 0.8  3.4]
 [ 1.4  3.4]
 [ 2.   3.4]
 [-1.   3.8]
 [-0.4  3.8]
 [ 0.2  3.8]
 [ 0.8  3.8]
 [ 1.4  3.8]
 [ 2.   3.8]
 [-1.   4.2]
 [-0.4  4.2]
 [ 0.2  4.2]
 [ 0.8  4.2]
 [ 1.4  4.2]
 [ 2.   4.2]
 [-1.   4.6]
 [-0.4  4.6]
 [ 0.2  4.6]
 [ 0.8  4.6]
 [ 1.4  4.6]
 [ 2.   4.6]
 [-1.   5. ]
 [-0.4  5. ]
 [ 0.2  5. ]
 [ 0.8  5. ]
 [ 1.4  5. ]
 [ 2.   5. ]]
'''