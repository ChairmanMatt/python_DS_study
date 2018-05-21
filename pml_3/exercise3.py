# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:53:31 2018

@author: mlu
"""

import pandas as pd
import numpy as np
data = pd.read_csv('data/data.csv')
X = np.array(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']])
y = np.array(data[['y']])
import sklearn.feature_selection as fs

select1 = fs.VarianceThreshold(10).fit_transform(X)
print(select1.shape)

print(select1[1])
print(X[1])

