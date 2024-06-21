###
### Demonstrating bias-variance with a simulated set of known data
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly) # This is the special ISLP library 
from ISLP.models import sklearn_sm, bs, ns
from pandas.api.types import is_numeric_dtype, is_object_dtype
def frame_stats(df):
    stats = pd.DataFrame(columns=df.columns, index=('Type', 'Min', 'Max', 'Mean'))
    cat_vars = []
    num_vars = []
    for col in df.columns:
        stats.loc['Type', col] = df[col].dtype
        if is_numeric_dtype(df[col].dtype):
            stats.loc['Min', col] = df[col].min()
            stats.loc['Max', col] = df[col].max()
            stats.loc['Mean', col] = df[col].mean()
            num_vars.append(col)
        if is_object_dtype(df[col].dtype):
            cat_vars.append(col)
    print(stats.transpose(), '\n')
    print('Rows = ' + str(df.shape[0]))
    print('Cols = ' + str(df.shape[1]))
    for col in cat_vars:
        print(col+': ', pd.unique(df[col]), '\n')
    return df.shape[0], df.shape[1], num_vars, cat_vars

Data = pd.read_csv('C:\\Users\\Mark\\source\\repos\\ISL_Course\\Quartic.csv') # Fix this later to use CWD   
frame_stats(Data)
Data.sort_values('X', inplace=True)
train_set, test_set = train_test_split(Data, test_size=0.10, shuffle=True, random_state=42)

def mse(Y, X, model):
    y_pred = model.predict(X)
    return np.mean((y_pred-Y)**2)

y_train = train_set['Y']
y_test = test_set['Y']
test_mse=[]
train_mse=[]
df = range(3,10)
for d in df:
    bspline = MS([bs('X', df=d, degree=3)]).fit(train_set)
    X_train = bspline.transform(train_set)
    X_test = bspline.transform(test_set)
    model = sm.OLS(y_train, X_train).fit()
    train_mse.append(mse(y_train, X_train, model))
    test_mse.append(mse(y_test, X_test, model))

fig = plt.figure(figsize=(10,5))
axx1 = fig.add_subplot(1,1,1)
#axx1.plot(Data['X'], model.fittedvalues, color='red')
axx1.plot(df, test_mse, label='Test')
axx1.set_xlabel('Degree')
axx1.set_ylabel('MSE')

#axx1.scatter(Data['X'], Data['Y'], s=9, alpha=0.6, color='grey')
axx1.plot(df, train_mse, label='Train')
axx1.legend();
plt.show()
