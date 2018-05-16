# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:13:13 2018

@author: mlu
"""

### your solution

import pandas as pd
import numpy as np

coeff = pd.DataFrame(columns=[["x1", "x2"]])

numtrials = 100
r2sum = 0

for i in range(0, numtrials): 
    arrx1 = pd.DataFrame(1000 * np.random.rand(30, 1) + 1000)
    arrx1.columns=["X1"]
    arrx2 = pd.DataFrame(5000 * np.random.rand(30, 1))
    arrx2.columns=["X2"]
    beta = [3, 5]
    
    arrx = arrx1.join(arrx2)
    arry = np.dot(arrx, beta) + (np.random.rand() * 5000)
    
    from sklearn import linear_model
    ols = linear_model.LinearRegression()
    
    ols.fit(arrx, arry)
    coeff.loc[i] = ols.coef_
    r2sum += ols.score(arrx, arry)
    
sums = coeff.sum()
means = sums/numtrials
r2sum = r2sum/numtrials
variance = ((coeff - means)**2).sum() / (numtrials-1)
