# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:47:34 2018

@author: mlu
"""
import pandas as pd
import numpy as np

from sklearn import naive_bayes
mnb = naive_bayes.MultinomialNB()


train = pd.read_csv('./data/spam_train.csv')
## separate the predictors and response in the training data set
x = np.array(train.iloc[:, 0:57])
y = np.ravel(train.iloc[:, 57:58])
train.head()

y == 'spam'
print("     spam:", np.sum(y == 'spam'))
print("None spam:", np.sum(y != 'spam'))

mnb.fit(x, y)
print("The score of multinomial naive bayes is: " + str(mnb.score(x, y)))