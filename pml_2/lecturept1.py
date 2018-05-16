# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:30:03 2018

@author: mlu
"""

### We first load the modules and data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
path_to_file = "./data/pml2tumor.csv"
data = pd.read_csv(path_to_file)
x_tm = data[["Size"]]
y_tm = data["Malignant"]
x_tm2 = np.copy(x_tm)
x_tm2[-3, 0] = 13
x_tm2[-1, 0] = 14

from sklearn import linear_model
logit_1 = linear_model.LogisticRegression()

logit_1.get_params()
# 'penality': 'l2' refers to Ridge regression

logit_1.set_params(C=1e4)
logit_1.fit(x_tm, y_tm)
print([logit_1.coef_, logit_1.intercept_])

print(logit_1.score(x_tm, y_tm))

logit_1.predict([[3], [4]])  # the nested list will be converted to 2D np.array automatically
logit_1.predict_proba([[3], [4]])

def data_1Dplot(x, y, xlabel=None, ylabel=None, labels=None, title=None):
    ## scatter plot the data
    plt.scatter(x, y, c=y, s=50, alpha=0.6)
    ## set labels
    if not xlabel is None:
        plt.xlabel(xlabel, size=12)
    if not ylabel is None:
        plt.ylabel(ylabel, size=12)
    ## set ticks for y
    y_ticks = np.unique(y)
    if not labels is None:
        plt.yticks(y_ticks, labels, \
                   rotation='vertical', size=12)
    ## set title
    if not title is None:
        plt.title(title, size=16)
        
def logistic_model_1Dplot(x, model, c="b"):
    x = np.array(x)
    num = 10000
    x = np.linspace(min(x), max(x), num=num).reshape(num,1)
    ## only plot the probability of prediction to be 1
    plt.plot(x, model.predict_proba(x)[:,1],
             ls='--', lw=2, c=c, label="Probability estimates")
    plt.plot(x, model.predict(x), lw=2, c=c, label="predictions")
    

## Plot the data points
plt.figure(figsize=(10, 6))
data_1Dplot(x_tm, y_tm, \
             xlabel="Size", ylabel="Malignant?", \
             labels=["No", "Yes"], \
             title="Tumor Size and Malignancy")
## Plot logistic model with original dataset
logistic_model_1Dplot(x_tm, logit_1)
## Set plot range
plt.axis([0,6,-0.2,1.2])
## Legend top-left corner
plt.legend(loc=2)
plt.show()

logit_2 = linear_model.LogisticRegression()
logit_2.set_params(C=1e4)
logit_2.fit(x_tm2, y_tm)
print([logit_2.coef_, logit_2.intercept_])

## Plot the data points
plt.figure(figsize=(10, 6))
data_1Dplot(x_tm2, y_tm, \
             xlabel="Size", ylabel="Malignant?", \
             labels=["No", "Yes"], \
             title="Tumor Size and Malignancy")
## Plot logistic model with original dataset
logistic_model_1Dplot(x_tm, logit_1)

## Plot logistic model with outlier
logistic_model_1Dplot(x_tm2, logit_2, c='g')   # set the color to green
## Set plot range
plt.axis([0,14,-.2,1.2])
## Legend bottom-right corner
plt.legend(loc=4)
plt.show()