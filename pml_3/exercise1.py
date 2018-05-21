# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:19:31 2018

@author: mlu
"""
import sklearn.model_selection as ms

from sklearn import datasets
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = ms.train_test_split(iris.data, iris.target, 
                                                       test_size=1.0/3, random_state=0)
print('Original: {}, {}'.format(iris.data.shape, iris.target.shape))
print('Training: {}, {}'.format(x_train.shape, y_train.shape))
print('Test: {}, {}'.format(x_test.shape, y_test.shape))

from sklearn import linear_model
logit = linear_model.LogisticRegression()
train_error = []
test_error = []

#### Your code here
for i in range(0, 5): 
    x_train, x_test, y_train, y_test = ms.train_test_split(iris.data, iris.target, test_size=1.0/3, random_state=i)

    logit.fit(x_train, y_train)

    train_error.append(sum((1.0/100) * ((y_train - logit.predict(x_train)) ** 2)))
    test_error.append(sum((1.0/50) * ((y_test - logit.predict(x_test)) ** 2)))
    
print("Training Error: " + str(train_error))
print("Test Error: " + str(test_error))

tst_err_mean = sum(test_error)/5
print("Test Error Mean: " + str(tst_err_mean))

testSD = ((1/5 * sum((test_error - tst_err_mean) ** 2))) ** 0.5
print("Test Error SD: " + str(testSD))


from sklearn import feature_selection as fs


iris.x = iris.data[:, 0:3]
iris.y = iris.data[:, 3]
best1 = fs.SelectKBest(fs.f_regression, k=1).fit_transform(iris.x, iris.y)