# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:16:06 2018

@author: mlu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

autos = pd.read_csv('data/Auto.csv')
mpg = autos["mpg"]
hp = autos["horsepower"].values.reshape(-1, 1)

from sklearn import linear_model

ols = linear_model.LinearRegression()

ols.fit(hp, mpg)


plt.plot(hp, ols.predict(hp), c='r', lw=1.5, label='Predicted relation')
plt.scatter(hp, mpg, c='k')
plt.xlabel('Power (HP)')
plt.ylabel('Fuel efficiency (MPG)')
plt.show()

print(ols.intercept_)
print(ols.coef_)
print(ols.score(hp, mpg))