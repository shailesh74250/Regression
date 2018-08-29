#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 05:10:53 2018

@author: shailesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('headbrain.csv')
dataset.head()

x = dataset.iloc[:, 2:3]
y = dataset.iloc[:, 3:4]

# splitting the data into training and test 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25,  random_state = 0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# predict the test result
y_pred = regression.predict(x_test)

# to see the relationship b/w the training data values
plt.scatter(x_train, y_train, c = 'red')
plt.show()

plt.plot(x_test, y_pred)
plt.scatter(x_test, y_test, c = 'red')
plt.xlabel('head size')
plt.ylabel('brain weight')