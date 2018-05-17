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

gnb = discriminant_analysis.LinearDiscriminantAnalysis()
lgsreg = linear_model.LogisticRegression(C=1e5)
print(lgsreg.get_params)

train = pd.read_csv('data/winequality-red.csv')
x = train.iloc[:, 0:10]
y = np.ravel(train.iloc[:, 11])

#threshold = 0.3
#x = feature_selection.VarianceThreshold(threshold).fit_transform(x, y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=43)

gnb.fit(x_train, y_train)
lgsreg.fit(x_train, y_train)


print("TRAIN: " + str(gnb.score(x_train, y_train)))
print("TEST: " + (str(gnb.score(x_test, y_test))))
print("\n")
print("LG TRAIN: " + str(lgsreg.score(x_train, y_train)))
print("LG TEST: " + (str(lgsreg.score(x_test, y_test))))