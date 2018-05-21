# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:29:47 2018

@author: mlu
"""

import pandas as pd
import numpy as np
data = pd.read_csv('data/data.csv')
X = np.array(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']])
y = np.array(data[['y']])
import sklearn.feature_selection as fs

select2 = fs.SelectKBest(fs.f_regression, 3).fit_transform(X, y.ravel())

print(select2[0])
print(X[0])