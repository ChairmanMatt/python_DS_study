# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:50:32 2018

@author: mlu
"""

import numpy as np
import pandas as pd
oj = pd.read_csv('data/OJ.csv')

data = oj.iloc[:, 1:]
target = oj.iloc[:, 0]

from sklearn import model_selection as ms
x_train, x_test, y_train, y_test = ms.train_test_split(data, target, test_size=1.0/2, random_state=0)

from sklearn import svm
svm_model = svm.SVC(kernel='poly')

svm_model.fit(x_train, y_train)


#print(svm_model.score(x_train, y_train))
#print(svm_model.score(x_test, y_test))

#train_scores = []
#test_scores = []
#c_values = []

#highscore_C = 10.985411419875572

#for i in np.logspace(start=-3, stop=3, base=10):
#    svm_model.set_params(C=i)
#    svm_model.fit(x_train, y_train)
#        
#    c_values.append(i)
#    train_scores.append(svm_model.score(x_train, y_test))
#    test_scores.append(svm_model.score(x_test, y_test))
    
#import matplotlib.pyplot as plt
#plt.scatter(c_values, train_scores)
#plt.scatter(c_values, test_scores)
#plt.show

#from sklearn import tree
##tree_model = tree.DecisionTreeClassifier()

##grid_para_tree = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 31)}
##grid_dt_search = ms.GridSearchCV(tree_model, param_grid=grid_para_tree, scoring='accuracy', cv=3, n_jobs=1)
##grid_dt_search.fit(x_train, y_train)
##print(grid_dt_search.best_params_)
##tree_model.set_params(criterion="gini", max_depth=4)
##tree_model.fit(x_train, y_train)
##feature_importance = np.array(list(zip(data.columns.values, tree_model.feature_importances_)), 
##                              dtype=[('feature', 'S10'), ('importance', 'float')])
##most_important = np.sort(feature_importance, order="importance")[::-1]
##for i in most_important[0:5]:
##    print(i)

grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 31),
    'n_estimators': range(10, 110, 10)
}

from sklearn import ensemble
forest_model = ensemble.RandomForestClassifier()

grid_rf_search = ms.GridSearchCV(forest_model, grid_para_forest, cv=3, n_jobs=1)
grid_rf_search.fit(x_train, y_train)
print(grid_rf_search.best_params_)
forest_model.set_params(criterion='gini', max_depth=3, n_estimators=80)
forest_model.fit(x_train, y_train)
print(forest_model.score(x_test, y_test))
feature_importance = np.array(list(zip(data.columns.values, forest_model.feature_importances_)), 
                              dtype=[('feature', 'S10'), ('importance', 'float')])
most_important = np.sort(feature_importance, order="importance")[::-1]
for i in most_important[0:5]:
    print(i)
