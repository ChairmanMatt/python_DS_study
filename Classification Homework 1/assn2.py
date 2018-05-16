# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:37:52 2018

@author: mlu
"""

import pandas as pd
import numpy as np
train = pd.read_csv('data/spam_train.csv')
test = pd.read_csv('data/spam_test.csv')

from sklearn import naive_bayes
mnb = naive_bayes.MultinomialNB()
import sklearn.feature_selection as skfs

sel = skfs.VarianceThreshold(threshold=(.8 * (1 - .8)))




numfeatures = 48
## separate the predictors and response in the training data set
x = pd.DataFrame(train.iloc[:, 0:numfeatures])
#sel.fit_transform(x)
x = x.drop(x.columns[[2, 4, 5, 8, 9, 11, 12, 13, 14, 17, 18, 20, 25, 26, 32, 35]], axis=1)
y = np.ravel(train.iloc[:, 57:58])


## separate the predictors and response in the test data set
x2 = pd.DataFrame(test.iloc[:, 0:numfeatures])
#sel.fit_transform(x2)
x2 = x2.drop(x2.columns[[2, 4, 5, 8, 9, 11, 12, 13, 14, 17, 18, 20, 25, 26, 32, 35]], axis=1)
y2 = np.ravel(test.iloc[:, 57:58])

## have a look at the training data set
train.head()
mnb.fit(x, y)
print("TRAIN: " + str(mnb.score(x, y)))
print("TEST: " + str(mnb.score(x2, y2)))


