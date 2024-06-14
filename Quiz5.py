# The labs for the ISL course are notebooks at https://github.com/intro-stat-learning/ISLP_labs 
# So I am doing them on Google Colab 
# The exercises, I am doing in Studio 

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

Quiz = pd.read_csv('F:\\Studio\\ISL_Course\\5.Py.1.csv') # The course data is at https://www.statlearning.com/resources-python 
print(Quiz.describe())       

# Here is normal OLS model 
y = Quiz['y']
design = MS(['X1', 'X2',])  
X = design.fit_transform(Quiz)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Here is OLS with sklearn wrapper (to enable use of CV) 
# model = sklearn_sm(sm.OLS, MS(['X1', 'X2'])) 
# X, Y = Quiz.drop(columns=['y']), Quiz['y']
# results = model.fit(X, Y)
# model_se = summarize(model.results_)['std err']
# print(model_se)

fig = plt.figure(figsize=(10,5))
axx1 = fig.add_subplot(1,2,1)
axx1.scatter(Quiz['X1'], results.fittedvalues, s=9)
axx1.set_xlabel('X1')
axx1.set_ylabel('Fitted')
axx1.scatter(Quiz['X1'], Quiz['y'], s=9)
# plt.show()

len = X.shape[0]
rng = np.random.default_rng(42)
se_accum = 0
iter = 1000
print('Standard bootstrap with all rows')
for i in range(iter):
    idx = rng.choice(X.index, size=len, replace=True) # Numpy way 
    # newX = X.sample(n=len, replace=True)   # Pandas way
    newX = X.iloc[idx]
    newY = y.iloc[idx]
    model = sm.OLS(newY, newX)
    results = model.fit()
    se = results.bse.iloc[1]
    se_accum += se
    # print(i, se)
print(se_accum/iter)

blocks = np.array_split(Quiz, 10) # Original df so that Y gets shuffled too
print('Block bootstrap with ten blocks')
se_accum = 0
for i in range(iter):
    # rng.shuffle(blocks) Shuffle assumes no replacement
    sampled_blocks = rng.choice(blocks, size=10, replace=True)
    newset = pd.DataFrame(columns=['X1', 'X2', 'y'])
    for i in range(sampled_blocks.shape[0]): # Choice returns extra dimension 
        block = pd.DataFrame(sampled_blocks[i,:,:], columns=['X1', 'X2', 'y'])
        newset = pd.concat([newset, block])
    y = newset['y']
    X = newset[['X1', 'X2']]
    X = sm.add_constant(X) # Tired of using MS 
    model = sm.OLS(y, X)
    results = model.fit()
    se = results.bse.iloc[1]
    se_accum += se
print(se_accum/iter)
# print(results.summary())