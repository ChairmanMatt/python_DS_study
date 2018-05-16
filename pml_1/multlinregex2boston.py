# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:59:55 2018

@author: mlu
"""

import numpy as np

from sklearn import datasets
boston = datasets.load_boston()
import pandas as pd
X_bos = pd.DataFrame(boston.data, columns=boston.feature_names)
y_bos = pd.Series(boston.target, name='MEDV')
## use head() to check the first few rows
X_bos.head()

from sklearn import linear_model
ols = linear_model.LinearRegression()

#print(X_bos.describe())
#print(X_bos["CHAS"].value_counts())
#print(X_bos["RAD"].value_counts())


df = X_bos.copy()
rad_dummy = pd.get_dummies(df["RAD"]).drop([1.0], 1)
rad_dummy.columns=["RAD_2.0", "RAD_3.0", "RAD_4.0", "RAD_5.0", "RAD_6.0", "RAD_7.0", "RAD_8.0", "RAD_24.0"]
df = df.drop("RAD", 1).join(rad_dummy)
df.head()
pd.DataFrame(df.columns, columns=['Column_Name'])

try:  # train_test_split was moved in 0.18.0
    from sklearn.model_selection import train_test_split
except:  # Following import works through 0.19 but outputs a warning in 0.18
    from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y_bos, test_size=0.3, random_state=42)

#### Your code here
ols.fit(X_train, y_train)
