import numpy as np

np.random.seed(1)
x = np.random.random(3)
print(x)

print(np.sum(x))
print(x.sum())

print(np.max(x))
print(x.max())

print(np.min(x))
print(x.min())

A = np.arange(1,16).reshape(3,-1)
print(A)
'''
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]]
'''

print(np.sum(A))
#120

#沿着行压缩，每列求和。重要，常用！
print(np.sum(A,axis=0))
#[18 21 24 27 30]

#沿着列压缩，每行求和。重要，常用！
print(np.sum(A,axis=1))
#[15 40 65]

print(np.prod(A+1))
#矩阵中所有元素的乘积  2004189184

v = np.array([10000,20000,20000,99990000])
print(np.mean(v))
#均值 25010000.0，均值对异常特别大或特别小的值影响大，例如被首富巨头拉高的居民平均收入

print(np.median(v))
#中位数 20000.0  中位数对异常大小值影响不大

print(np.percentile(v,q=25))
#百分位，例如q=50的数值是中位数，q=100的数值是max，q=0的数值是min。一般看q=0,25,50,75,100的分布情况
#17500.0


s = np.random.normal(loc=0, scale=1, size=100000)
print(np.mean(s))
#均值loc  0.0029501121674453917

print(np.var(s))
#方差scale  1.0004732097808802

print(np.std(s))
#标准差 1.0002365769061239






