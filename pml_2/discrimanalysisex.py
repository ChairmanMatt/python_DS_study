# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:19:52 2018

@author: mlu
"""

import numpy as np
import pandas as pd

#### Load the data if you haven't done so

from sklearn import datasets
iris = datasets.load_iris()
pair = (2, 3)

xlabel = iris.feature_names[pair[0]]
ylabel = iris.feature_names[pair[1]]

iris_x = iris.data[:, pair]
iris_y = iris.target

from sklearn import discriminant_analysis
gnb = discriminant_analysis.LinearDiscriminantAnalysis(C=1e4)

#### Your code here

gnb.fit(iris_x, iris_y)
print(gnb.score(iris_x, iris_y))

x_new = pd.DataFrame([2.8, 5.6], [3.2, 6.7]).values.reshape(2, -1)

print(gnb.predict(x_new))
print(gnb.predict_proba(x_new))