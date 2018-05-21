# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:50:32 2018

@author: mlu
"""

import pandas as pd
oj = pd.read_csv('data/OJ.csv')
#oj.head()

data = oj.iloc[:, 1:]
target = oj.iloc[:, 0]

from sklearn import model_selection as ms

x_train, x_test, y_train, y_test = ms.train_test_split(data, target, test_size=1.0/2, random_state=0)

