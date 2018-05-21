# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:56:05 2018

@author: mlu
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms

n = 500
np.random.seed(1)
X = np.random.randn(n, 2)
y = np.ones(n)
y[X[:, 0] + X[:, 1] > 0] = 0
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn import discriminant_analysis
model = discriminant_analysis.LinearDiscriminantAnalysis()

## use train_test_split to split the dataset into training and test datasets

## fit a LDA model with x_train, y_train
## Train and test error will vary with random split

x_train, x_test, y_train, y_test = ms.train_test_split(X, y, test_size=1.0/2, random_state=18)

model.fit(x_train, y_train)
train_error = 1/250 * sum((y_train - model.predict(x_train)) ** 2)
test_error = 1/250 * sum((y_test - model.predict(x_test)) ** 2)

print("TRAIN ERROR: " + str(train_error))
print("TEST ERROR: " + str(test_error))


train_err_list = []
test_err_list = []

for train, test in ms.KFold(n_splits=5, random_state=4).split(X) :
    x_2train, x_2test, y_2train, y_2test = X[train], X[test], y[train], y[test]
    model.fit(x_2train, y_2train)
    train_err_list.append(1/250 * sum((y_2train - model.predict(x_2train)) ** 2))
    test_err_list.append(sum((1.0/50) * ((y_2test - model.predict(x_2test)) ** 2)))

print("Train error: " + str(train_err_list))
print("Test error: " + str(test_err_list))


scores = ms.cross_val_score(estimator=model, X=X, y=y, cv=ms.KFold(n_splits=5, random_state=0))
errors = 1-scores