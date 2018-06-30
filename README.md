学习传统机器学习领域知识时，边学习边编码实践，深入理解传统机器学习算法底层数学原理后，仿scikit-learn封装函数实现传统机器学习算法类，最后实践调用scikit-learn对比。   
含随机拆分样本为训练集和测试集、数据集标准化、拟合fit、预测predict，效果评估r2、KNN分类和回归算法、线性回归算法、梯度下降法(非机器学习算法)、PCA算法、

## NumpyMatplotlib

   传统机器学习所需的Numpy和Matploblib的一些基础尝试练习。

## KNN 
- KNNStepByStep.py：  
没考虑任何归一化等复杂处理，最简单的方式实现KNN分类算法主流程：手工造样本数据和标识数据，加入一个test点，画散点图看位置归属。然后按照KNN算法思路，逐步实现KNN分类算法预测。验证与查看散点图结果是一致的。
- KNNFunction.py  
函数封装KNN分类算法，在KNNStepByStep.py执行验证

- KNNScikitLearnTest.py：  
 用自己造的数据测试验证scikit-learn的封装KNN分类算法 

- KNNClassfier.py  
仿scikit-learn封装KNN算法，写KNN分类算法实现类  

- KNNFunction.py  
自己写了个KNN分类算法，做了函数封装

- KNNIris.py  
引入scikit-learn中的150个鸢尾花数据，将数据随机拆分为120个训练样本和30个测试样本，数据归一化处理，训练样本fit，然后测试样本predict，并将预测值与30个测试标签实际值对比，计算查准率。

- KNNDigits.py   
引入scikit-learn中的1797个手写数字8*8像素数据，将数据随机拆分为训练样本和测试样本，数据归一化处理，训练样本fit，然后测试样本predict，并将预测值与测试标签实际值对比，计算查准率。  
测试交叉验证和基于交叉验证的网格搜索，做KNN的超参数调优。

-  KNNGridSearch.py  
引入scikit-learn中的1797个手写数字8*8像素数据，将数据随机拆分为训练样本和测试样本，自己尝试编写了网格搜索超参数调优，然后又用scikit-learn做KNN分类的超参数调优。调优结果超参数和调优结果查看

## Scaler
- NormalizationScalingTest.py  
自己写了个最值归一化方法测试。

- StandardizationScalingTest.py  
自己写了个均值方差归一化方法测试。

- StandardScaler.py  
仿scikit-learn封装均值方差归一化算法，写了个算法实现类 

## Utils 
- TrainTestSplitFunction.py  
自己封装了一个函数：将样本数据，按比例拆分为训练样本和测试样本。

- AccuracyFunction.py  
KNN分类算法时，封装了计算测试样本预测结果，与测试样本真实结果的查准率的accuracy_score函数。  
线性回归算法时，封装了mse、rmse、mae、R方损失函数。

-  PlotLearningCurve.py  
1.编写了一个画学习曲线的函数plot_decision_boundary：随训练样本数的增多，训练数据集的预测RMSE和测试数据集的预测RMSE的趋势图。多项式回归做了测试。   
2.在plot_decision_boundary基础上，针对SVM，在决策边界上下补充2条支撑向量显示

-  PlotDecisionBoundary.py  
将x和y坐标栅格化的点组成一个新的待预测矩阵X_new，预测值Y_predict，绘制等高线，形成算法决策边界图

-  Evaluate2Class.py  
2分类算法评判标准，自己实现了如下函数：TN、FP、FN、TP、混淆矩阵、精准率、召回率、F1评分、TPR、FPR。


## LinearRegression 
- SimpleLinearRegressionStepByStep.py  
用最简单的方法逐步尝试实现一维线性回归，并分别调用了自己封装的函数、scikit-learn原生的MSE、RMSE、MAE、R方损失函数计算误差

- SimpleLinearRegressionClassfier1.py  
封装一维逻辑线程回归算法

- SimpleLinearRegressionClassfier2.py  
改良上述一维逻辑线性回归算法，提升性能
使用向量化计算，使用向量点乘，对应 i 值的数值相乘，然后相加成1个数，性能比for循环高很多

- LinearRegressionClassfier.py  
仿scikit-learn封装，自己编写了一个线性回归类，具备拟合(XXfit)、预测(predict)、效果评估(R方)功能。其中：   
**拟合函数fit_normal是使用正规方程解拟合**(最小损失函数的正规方程解)。缺点：时间复杂度高,O(n的3次方，算法优化后也有n的2.4次方)，不推荐使用，推荐使用梯度下降法)。  
**拟合函数fit_gd是使用批量梯度下降法拟合**。因为正规方程解通过矩阵乘法和求逆运算来计算参数。当变量很多的时候计算量会非常大。 #使用批量梯度下降法求解最小损失函数，比常规方程解好。
#eta是每次求解移动的步长，max_iter是梯度下降最大迭代次数  
**拟合函数fit_sgd是使用随机梯度下降法拟合**。因为批量梯度下降法随样本数m增大，计算量还是增大的，所以考虑“随机梯度下降法”，进一步降低样本数m增大对计算法增大的影响。max_iter的含义变为需要对样本遍历几遍，例如样本数为m，max_iter = 5的含义是，遍历m*5次。#批量梯度下降法每次迭代都用所有样本，快速收敛，稳定，但性能不高。 #随机梯度下降法每次用一个样本调整参数，降低计算量，计算量不再受样本数量影响，逐渐逼近，效率高,随机还有可能跳出局部最优解。        #但每次减小效果的不像批量梯度下降法那么好(稳定)，但经过验证还是能找到最小损失函数的。        #如果步长固定，随机梯度下降法最后可能会在最小损失函数附近来回波动却找不到，所以步长需要开始大，越向后越小。 #最简单的想法是步长eta = t0/(当前迭代次数 + t1)--模拟退火思想。        #a和b是随机梯度下降法的超参数，为避免算法最开始步长下降太大，经验值t0=5,t1=50。        #随机梯度下降法也不用判断两次损失函数之间的最小差值epsilon了--因为不是绝对梯度，不能保证一直下降。
-  LinearRegressionBoston.py  
基于波士顿房价数据：  
1.测试验证了自己写的LinearRegressionClassfier.py的正规方程解算法、批量梯度下降法、随机梯度下降法和R方效果评估。  
2.测试验证了scikit-learn的线性回归算法的批量梯度下降法、随机梯度下降法和R方效果评估。  
3.测试验证了scikit-learn的KNN回归算法和R方效果评估。

##  GradientDescent
-  GradientDescentFirstTry.py  
对一个简单的线性关系，基于梯度下降法的数学思想，求导方式编写基础的验证梯度下降法，并画图验证。
-  GradientDescentDebug.py   
在每个theta点两边极近相等距离取2个点，这两个点直线斜率约等于其导数。用途：该方法是通用的，适用于所有函数求梯度，不限于线性回归。但性能不好，可用于测试验证看自己求导的结果于此是否一致。

##  PCA
-  PCAStepByStep.py  
造了有线性关系+噪声的100个样本，2个特征的数据集，基于梯度上升法，实现了PCA降维，并画图验证。  
调用自己写的PCA类尝试。
调用scikit-learn的PCA类尝试。
inverse恢复与原数据集对比损失。

-  PCAClassfier.py  
仿scikit-learn封装了PCA的拟合、PCA转换transform，逆转换函数。

-  PCADigits.py  
基于手写识别数据，先PCA降维再KNN分类，体验PCA降维小损失带来处理性能极大提升。  
画图体验PCA降维 和 损失的曲线。  
10维数据降维到2维,方便画图查看数据

##  PolynomialRegression
-  PolynomialRegressionStepByStep    
自己造了个非线性二次幂数据集，分别用：  
1.线性回归拟合；  
2.自己加二次幂样本后拟合；  
3.调用scikit-learn的PolynomialFeatures(degree=2)补充加二次幂样本后拟合。   
4.调用自己写的多项式回归函数PolynomialRegressionFunction拟合  
最后分别输出参数和截距与原数据对比、画图对比查看。

-  PolynomialRidgeLassoRegression.py  
1.使用pipelline管道流程组装样本增维PolynomialFeatures(degree参数)、样本标准化StandardScaler()、线性回归LinearRegression() 为PolynomialRegression函数。  
2.使用pipelline管道流程组装样本增维PolynomialFeatures(degree参数)、样本标准化StandardScaler()、模型正则化-岭回归Ridge(alpha=alpha)避免过拟合， 为RidgeRegression函数。  
3.使用pipelline管道流程组装样本增维PolynomialFeatures(degree参数)、样本标准化StandardScaler()、模型正则化-LASSO回归Lasso(alpha=alpha)避免过拟合，为LassoRegression函数。

-  PolynomialRidgeLassoTest.py  
构造一阶线性数据集，分别使用20维多项式回归、岭回归和LASSO回归fit，并绘到一张图上，验证岭回归和LASSO回归这两种模型正则法对过拟合的优化


-  LearningCurveTest.py
调用utils中自己编写的学习曲线函数PlotLearningCurve，对比分析线性回归和2阶多项式回归的学习曲线。

##  LogisticRegression
-  LogisticRegressionClassfier.py  
根据逻辑回归的公式推导，在自己写的线性回归类基础上，改写的逻辑线性回归类，含内部sigmoid、 批量梯度下降拟合(含损失函数J和对J的求导)、预测概率、预测函数等。

-  LogisticRegressionIris.py  
将鸢尾花数据取前2类，因为逻辑回归默认只能处理二分类问题，所以只取鸢尾花数据集前2种分类；便于画图只取前2个特征。调用LogisticRegressionClassfier.py 预测，并绘制决策边界。  
将鸢尾花数据便于画图只取前2个特征，调用KNN决策算法预测，并绘制决策边界。  
二分类转变为多分类OVO和OVR，scikit-learn分别提供了2种方法。最后绘制决策边界


-  PolynomialLogisticRegression.py  
基于pipeline将PolynomialFeatures、StandardScaler、自己写的线性逻辑回归/scikitlearn的线性逻辑回归(引入超参数C和模型正则化penalty，避免过拟合)，组装多项式逻辑回归函数

-  PolynomialLogisticRegression.py  
自己造了份带噪音的抛物线曲线样本，分别用自己写的多项式多项式逻辑回归函数、scikitlearn的线性逻辑回归、scikitlearn的多项式逻辑回归拟合、评分、绘制决策边界。

##  ClassfierEvaluate
-  Skewed2ClassEvaluateTest.py  
将数字数据集改写为一个极度有偏的2分类数据集，然后针对该数据集分别调用Evaluate2Class.py和scikit-learn中的2分类算法评判标准函数：TN、FP、FN、TP、混淆矩阵、精准率、召回率、F1评分、TPR、FPR尝试，并绘制相应的如PR、ROC曲线观测关键点。

-  MultiClassEvaluate.py  
基于10分类的数字数据集，调用scikit-learn中精准率、混淆矩阵查看，并基于混淆矩阵输出误差矩阵，灰度绘制。

## SVM
-  LinearSVC2Iris.py 
便于画图，取鸢尾花数据前2类；前2个特征，标准化数据后，调用scikit-learn中的线性SVM算法LinearSVC拟合，调整超参数C为大值(Hard Margin SVM)、小值(Soft Margin SVM)，绘制决策边界体验支撑向量区间的变化。

-  SVCFunction.py  
分别使用传统多项式的线性SVM 和 使用多项式核函数的SVM的分类方式，封装非线性SVM分类算法。

-  PolyNomialSVCTest.py  
使用scikit-learn中的moon加噪数据集。分别调用SVCFunction.py中的传统多项式的线性SVM分类和多项式核函数的SVM的分类方式拟合数据，分别绘制决策边界体验。

-  SVRFunction.py  
SVM算法也可以作为回归算法使用。封装了一个标准化数据+SVM线性回归算法。体验超参数epsilon。

-  SVRBoston.py  
基于波士顿数据集，调用封装的SVRFunction.py的SVM回归算法，拟合数据集查看评分。 

##  DecisionTree  
-  DecisionTreeClassfierFunction.py  
自己写了个在每个特征维度dim，每个样本m排序value值mean二分的方法，取信息熵或基尼系数最小值的决策树函数


-  DecisionTreeEntropyClassfierIris.py  
基于鸢尾花数据集，分别调用scikit-learn和自己写的DecisionTreeClassfierFunction.py函数，使用信息熵超参数，做分类，然后绘制决策边界查看

-  DecisionTreeGiniClassfierIris.py  
基于鸢尾花数据集，分别调用scikit-learn和自己写的DecisionTreeClassfierFunction.py函数，使用基尼系数超参数，做分类，然后绘制决策边界查看

-  DecisionTreeParamClassfierMoons.py
基于双月加噪数据集，调整决策树的各种超参数，绘制决策边界查看分类拟合效果

-  DecisionTreeRegressionBoston.py  
基于波士顿房价数据，调用scikit-learn的决策树回归拟合，查看评分

##  EnsembleLearning  
-  VotingMoons.py  
1.基于双月加噪数据集，自己写个简单的让3种算法投票集成学习。  
2.调用scikit-learn的集成学习：投票hard voting。  
3.调用scikit-learn的集成学习：概率权重投票soft voting。soft voting比hard votng更合理，soft voting要求集合的每一个模型都能估计概率predict_proba。

-  BaggingMoons.py  
放回取样Bagging学习：  
1.样本产生差异化的方法1：样本随机放回取样。调用scikit-learn的集成学习：放回取样bagging(bootstrap=True)。  
2.样本产生差异化的方法1：样本随机放回取样改良：更合理的最大化利用有限的样本数据的方法。调用scikit-learn的集成学习：放回取样bagging(bootstrap=True,oob_score=True)。  
3.样本产生差异化的方法2：不随机取样本，而是随机取特征random_subspaces。  
4.样本产生差异化的方法3：既随机取样本，又随机取特征random_patches。

-  RandomForestMoons.py  
scikit-learn随机森林分类器 和 Extra Trees分类器。

-  BoostingMoons.py  
1.调用scikit-learn的集成学习：Ada Boosting。Ada Boosting，第一个算法fit后，会降低大部分拟合样本点的权重，然后下个算法再次拟合。  
2.调用scikit-learn的集成学习：Gradient Boosting。Gradient Boosting，训练一个模型m1，产生错误e1；针对e1训练第2个模型m2,产生错误e2；针对e2训练第3个模型m3,产生错误e3...最终预测结果是m1+m2+m3+...Gradient Boosting自身就是以决策树为基础的。