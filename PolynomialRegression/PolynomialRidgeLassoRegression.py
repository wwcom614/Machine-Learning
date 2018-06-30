from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
def PolynomialRegression(degree):
    #将样本增维、标准化、线性拟合综合到一起，使用pipeline
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

from sklearn.linear_model import Ridge
#岭回归，模型正则化方法之一，用于消除过拟合，降低方差。正则化是1~n的theta的平方和 * alpha /2
# 岭回归引入新的超参数alpha，alpha越大，曲线越平滑。alpha超级大时，岭回归是一根和X轴平行的直线
def RidgeRegression(degree, alpha):
    #将样本增维、标准化、线性拟合综合到一起，使用pipeline
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])

from sklearn.linear_model import Lasso
#LASSO回归，模型正则化方法之一，用于消除过拟合，降低方差。正则化是1~n的theta的绝对值和 * alpha
# LASSO回归引入新的超参数alpha，alpha越大，曲线越平滑。
# alpha=0.1时，LASSO回归近似一条斜线。岭回归始终是曲线，而LASSO回归近似为倾斜的直线
# alpha=1时，LASSO回归是一根和X轴平行的直线
#LASSO：最小绝对值紧缩和选择运算符
#LASSO趋向于使得一部分theta值变为0，所以可以用于特征选择  Selection Operator(SO)
#LASS0的参数趋向于-1,1,0的方式，有可能将有用的特征成为0去掉，所以一般岭回归效果更好，建议优先选择岭回归。
# 但如果theta值大，岭回归的平方就可能不占优势了。此外，如果theta数量过多，岭回归计算量大，此时建议使用弹性网络
def LassoRegression(degree, alpha):
    #将样本增维、标准化、线性拟合综合到一起，使用pipeline
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))
    ])

