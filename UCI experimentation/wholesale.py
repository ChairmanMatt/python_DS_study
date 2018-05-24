# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:19:20 2018

@author: mlu
"""


import pandas as pd
import numpy as np

#from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import feature_selection

linreg_model = linear_model.LinearRegression()

train = pd.read_csv('data/Wholesale customers data.csv')
x = train.iloc[:, 3:7]
y = train.iloc[:, 2]

#x = feature_selection.SelectKBest(feature_selection.chi2, 20).fit_transform(x, y)
#x = feature_selection.RFE(linear_model.LinearRegression(), 20).fit_transform(x, y)

linreg_model.fit(x, y)

print(linreg_model.score(x, y))

#import matplotlib.pyplot as plt
#fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
#x.plot(ax=axes[0], kind='scatter', x='WEIGHT', y='MPG')
#x.plot(ax=axes[1], kind='scatter', x='HORSEPOWER', y='MPG')
#x.plot(ax=axes[2], kind='scatter', x='DISPLACEMENT', y='MPG')
#x.plot(ax=axes[3], kind='scatter', x='MODELYEAR', y='MPG')
#plt.show()