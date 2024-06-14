import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly) # This is the special ISLP library 
from ISLP.models import sklearn_sm
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

Quiz = pd.read_csv('F:\\Studio\\ISL_Course\\7.Py.1.csv') # Data for quizzes is linked from each quiz page  
print(Quiz.describe())       
  
# Here is normal OLS model 
y = Quiz['y']
Quiz['X2'] = Quiz['x']**2 
design = MS(['x', 'X2'])  
X = design.fit_transform(Quiz)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

fig = plt.figure(figsize=(10,5))
axx1 = fig.add_subplot(1,2,1)
axx1.scatter(Quiz['x'], results.fittedvalues, s=9)
axx1.set_xlabel('X')
axx1.set_ylabel('Y')
axx1.scatter(Quiz['x'], Quiz['y'], s=9)
plt.show()
