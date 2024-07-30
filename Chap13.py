##
## Chap 13 exercise 7
##
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests as mult_test
from ISLP.models import (ModelSpec as MS, summarize, poly) # This is the special ISLP library 
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

Data = pd.read_csv('F:\\Studio\\ISL_Course\\Datasets\\Carseats.csv') # Data for quizzes is linked from each quiz page  
frame_stats(Data)      
num_vars = np.array(['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education'])
p_values = []
for col in num_vars:
    y = Data['Sales']
    design = MS([col])
    X = design.fit_transform(Data)
    results = sm.OLS(y, X).fit()
    p = results.pvalues[1]
    p_values.append(p)
    
print(p_values) # 7a

p_values = np.array(p_values) 
reject = (p_values < 0.05)
print(num_vars[reject]) # 7b

reject = (p_values < 0.05/7)
print(num_vars[reject]) #7c

fdr = mult_test(p_values, alpha = 0.2, method = 'fdr_bh')[0]
print(num_vars[fdr]) #7d

