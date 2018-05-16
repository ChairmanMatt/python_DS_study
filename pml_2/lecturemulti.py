# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:43:24 2018

@author: mlu
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

path_to_file = "./data/pml2admission.csv"
data = pd.read_csv(path_to_file)
data.head()

## Data
scores = data[["Test_1", "Test_2"]]
decision = data["Decision"]

## Import linear_model from sklearn if you haven't
## Initialize a LogisticRegression instance with C=10e4
decision_logit = linear_model.LogisticRegression(C=1e4)

#### Your code here
decision_logit.fit(scores, decision)

print(decision_logit.score(scores, decision))

print(decision_logit.intercept_)
print(decision_logit.coef_)