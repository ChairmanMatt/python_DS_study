# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:31:14 2018

@author: mlu
"""

import pandas as pd
import numpy as np

#from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn import linear_model
from sklearn import feature_selection

gnb = discriminant_analysis.QuadraticDiscriminantAnalysis()
lgsreg = linear_model.LogisticRegression(C=1e5)
linreg = linear_model.LinearRegression()
print(lgsreg.get_params)

train = pd.read_csv('data/winequality-red.csv')
#train = train.loc[train["quality"] <= 9]

x = train.iloc[:, 0:11]
y = np.ravel(train.iloc[:, 11])


#threshold = 0.3
#x = feature_selection.VarianceThreshold(threshold).fit_transform(x, y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=43)

#gnb.fit(x_train, y_train)
lgsreg.fit(x_train, y_train)
linreg.fit(x_train, y_train)

print("LN TRAIN: " + str(linreg.score(x_train, y_train)))
print("LN TEST: " + (str(linreg.score(x_test, y_test))))
print("\n")
print("LG TRAIN: " + str(lgsreg.score(x_train, y_train)))
print("LG TEST: " + (str(lgsreg.score(x_test, y_test))))

new_wine = [(11.3,0.62,0.67,5.2,0.086,6,19,0.9988,3.22,0.69,13.4)]
new_wine = pd.DataFrame.from_records(new_wine)
print(linreg.predict(new_wine))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16))
train.plot(ax=axes[0][0], kind='scatter', x='fixed acidity', y='quality')
train.plot(ax=axes[0][1], kind='scatter', x='volatile acidity', y='quality')
train.plot(ax=axes[0][2], kind='scatter', x='citric acid', y='quality')
train.plot(ax=axes[1][0], kind='scatter', x='residual sugar', y='quality')
train.plot(ax=axes[1][1], kind='scatter', x='chlorides', y='quality')
train.plot(ax=axes[1][2], kind='scatter', x='free sulfur dioxide', y='quality')
train.plot(ax=axes[2][0], kind='scatter', x='total sulfur dioxide', y='quality')
train.plot(ax=axes[2][1], kind='scatter', x='density', y='quality')
train.plot(ax=axes[2][2], kind='scatter', x='pH', y='quality')
train.plot(ax=axes[3][0], kind='scatter', x='sulphates', y='quality')
train.plot(ax=axes[3][1], kind='scatter', x='alcohol', y='quality')
