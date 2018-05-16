# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:42:23 2018

@author: mlu
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import datasets


iris = datasets.load_iris()
pair = (2, 3)

xlabel = iris.feature_names[pair[0]]
ylabel = iris.feature_names[pair[1]]

iris_x = iris.data[:,pair]
iris_y = iris.target

iris_logit = linear_model.LogisticRegression(C=1e4)
#print(i_ls.get_params())

iris_logit.fit(iris_x, iris_y)

print(iris_logit.score(iris_x, iris_y))
print(iris_logit.coef_)
print(iris_logit.intercept_)