import numpy as np
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

## 数据归一化
# 幅度缩放
scaled_X = preprocessing.scale(X)
# 归一化
normalized_X = preprocessing.normalize(X)
# 标准化
standardized_X = preprocessing.scale(X)

## 特征选择
model = ExtraTreesClassifier()
model.fit(X, y)
# 特征重要度
print('Feature importances:')
print(model.feature_importances_)

## 建模与评估
model = LogisticRegression(max_iter=1000)  # 增加迭代次数以确保收敛
model.fit(X, y)
print('MODEL')
print(model)

## 预测
expected = y
predicted = model.predict(X)

## 输出评估结果
print('RESULT')
print(metrics.classification_report(expected, predicted))
print('CONFUSION MATRIX')
print(metrics.confusion_matrix(expected, predicted))

## 超参数调优
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X, y)
print('Best parameters found:')
print(grid_search.best_params_)
