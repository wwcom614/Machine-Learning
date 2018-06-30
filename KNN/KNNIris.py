#加载scikit-learn中的鸢尾花数据集，150条记录
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

X = iris.data
y = iris.target

#使用scikit-learn的split方法来拆分为训练数据集和测试数据集。 test_size默认值0.2。random_state是seed，调试使用
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#数据归一化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X_train)
print("训练集的均值:",standardScaler.mean_)
#标准差
print("训练集的标准差:",standardScaler.scale_)
X_train_standard = standardScaler.transform(X_train)
print("归一化训练集：",X_train_standard)

X_test_standard = standardScaler.transform(X_test)
print("归一化测试集：",X_test_standard)

#将scikit-learn中的鸢尾花数据集随机打散，拆分为训练数据集120条，测试数据集30条
#from Utils.TrainTestSplitFunction import train_test_split

#test_ratio = 0.2
#X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)

#使用scikit-learn的KNN类来分类
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)

#使用自己写的KNN类来分类
#from KNN.KNNClassfier import KNNClassfier
#knn_clf = KNNClassfier(k = 3)


knn_clf.fit(X_train_standard, y_train)
y_predict = knn_clf.predict(X_test_standard)
score = knn_clf.score(X_test_standard, y_test)

#score = sum(y_predict == y_test)/len(y_test) * 100
print(score)





