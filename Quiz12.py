##
## A lot of R crap in this exercise, like rbind and get_rdataset, that doesn't work
##
from sklearn import linear_model as lm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pyreadr  

#Data = get_rdataset('10.RData', package='datasets').data
Data = pyreadr.read_r('10.RData')
x = Data['x']
y = Data['y']
x_test = Data['x.test'] # Test set has 1000 cases
y_test = Data['y.test']
x_big = pd.concat([x, x_test])
print(x_big.head())
print(x_big.shape)

X = StandardScaler().fit_transform(x_big)
pca = PCA().fit(X)
print(pca.explained_variance_ratio_[:5])
print(pca.explained_variance_ratio_[:5].sum()) # This is question 1

x_new = PCA(n_components=5).fit_transform(X)
model = lm.LinearRegression().fit(x_new[:300,:], y) # Fit on the original 300 rows
y_pred = model.predict(x_new[300:,:]) # Predict on the bottom 1000
error = (y_pred - y_test)**2 
print(error.sum() / y_test.shape[0]) # This is question 2

model2 = lm.LinearRegression().fit(x, y)
y_pred = model2.predict(x_test)
error = (y_pred - y_test)**2 
print(error.sum() / y_test.shape[0]) # This is question 3