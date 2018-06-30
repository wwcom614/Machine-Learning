from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from LogisticRegression.LogisticRegressionClassfier import LogisticRegressionClassfier

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('logistic_reg', LogisticRegressionClassfier())
    ])

from sklearn.linear_model import LogisticRegression
def SKLPolynomialLogisticRegression(degree, C, penalty='l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('skl_logistic_reg', LogisticRegression(C=C, penalty=penalty))
    ])