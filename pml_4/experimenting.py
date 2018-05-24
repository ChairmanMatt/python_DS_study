# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:55:52 2018

@author: mlu
"""

from sklearn import svm
svm_model = svm.SVC(kernel='poly', C=1e5, degree=1)

import numpy as np
from sklearn import datasets
## prepare data
iris = datasets.load_iris()
index = range(100)
iris.x = iris.data[index, :]
iris.y = iris.target[index]
## fit
svm_model.fit(iris.x[:, 0:2], iris.y)

from sklearn import model_selection
grid_para_svm = [
    {'C': [1, 10, 100, 1000],
     'kernel': ['poly'],
     'degree': [1, 2, 3]},
    {'C': [1, 10, 100, 1000],
     'gamma': [0.001, 0.0001],
     'kernel': ['rbf']}
]

grid_search_svm = model_selection.GridSearchCV(svm_model, grid_para_svm, scoring='accuracy', cv=3, n_jobs=-1)
grid_search_svm.fit(iris.data, iris.target)

print(grid_search_svm.best_params_)