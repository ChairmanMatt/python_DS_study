# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:37:52 2018

@author: mlu
"""

import pandas as pd
import numpy as np
train = pd.read_csv('data/spam_train.csv')
test = pd.read_csv('data/spam_test.csv')

#from sklearn import naive_bayes
#mnb = naive_bayes.MultinomialNB()
from sklearn import discriminant_analysis
lda = discriminant_analysis.LinearDiscriminantAnalysis()

import sklearn.feature_selection as skfs

#skb = skfs.SelectKBest(skfs.chi2)

numfeatures = 57
## separate the predictors and response in the training data set
x = pd.DataFrame(train.iloc[:, 0:numfeatures])
y = np.ravel(train.iloc[:, 57:58])
#x = skb.fit_transform(x, y)


## separate the predictors and response in the test data set
x2 = pd.DataFrame(test.iloc[:, 0:numfeatures])
y2 = np.ravel(test.iloc[:, 57:58])
#x2 = skb.transform(x2)
   
#mnb.fit(x, y)

lda.fit(x, y)

#print("MNB TRAIN: " + str(mnb.score(x, y)))
#print("MNB TEST: " + str(mnb.score(x2, y2)))

print("LDA TRAIN: " + str(lda.score(x, y)))
print("LDA TEST: " + str(lda.score(x2, y2)) + "\n")
