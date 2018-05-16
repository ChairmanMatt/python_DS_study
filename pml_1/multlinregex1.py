# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:32:53 2018

@author: mlu
"""

## load data into pandas dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

adver = pd.read_csv('data/adver.csv', index_col=0)
adver.head()

## Visualizing data, shown are Sales vs TV, Radio and Newspaper
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
adver.plot(ax=axes[0], kind='scatter', x='TV', y='Sales')
adver.plot(ax=axes[1], kind='scatter', x='Radio', y='Sales')
adver.plot(ax=axes[2], kind='scatter', x='Newspaper', y='Sales')
plt.show()


x = adver[["TV", "Radio", "Newspaper"]]
y = adver["Sales"]

from sklearn import linear_model
ols = linear_model.LinearRegression()

ols.fit(x, y)

print("Intercept: " + str(np.round(ols.intercept_, 6)))
print("Coefficients: " + str(np.round(ols.coef_, 6)))

print("R^2: " + str(np.round(ols.score(x, y), 6)))

print("Predicted sales: " + str(np.round(ols.predict([[50, 100, 30]])[0], 6)))