# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:44:20 2018

@author: mlu
"""
import pandas as pd
from numpy import arrange, array, linalg
import matplotlib.pyplot as plt

autos = pd.read_csv('data/Auto.csv')
mpg = autos["mpg"]
hp = autos["horsepower"].values.reshape(-1, 1)

slope, intercept, r_value, p_value, std_err = stats.linregress