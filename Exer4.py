# The labs for the ISL course are notebooks at https://github.com/intro-stat-learning/ISLP_labs 
# So I am doing them on Google Colab 
# The exercises, I am doing in Studio 

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly) # This is the special ISLP library 
from sklearn.linear_model import LogisticRegression
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
    return num_vars, cat_vars
def crossplot(df):
    cols = df.shape[1]
    pos = 1
    dots = 5
    fig = plt.figure(figsize=(cols,cols))

    for columnA in df:
        for columnB in df:
            ax = fig.add_subplot(cols,cols,pos)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            if columnA.title == columnB.title:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.text(0.5, 0.5, df[columnA].name, fontsize=12, ha='center', va='center')
            else:
                ax.scatter(df[columnB], df[columnA], s=dots)
                # Organized so that each label is X for plots in its column 
            pos +=1
    plt.tight_layout()        
    plt.show()        
def heatmap(df):
    correlations = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.show()


stocks = load_data('Weekly') # The course data is at https://www.statlearning.com/resources-python 
num_vars, cat_vars = frame_stats(stocks)  
# crossplot(stocks)
# print(stocks.corr(numeric_only=True).round(decimals=2))
# heatmap(stocks[num_vars])

allvars = stocks.columns.drop(['Today', 'Direction', 'Year'])

# This is the GLM way we're supposed to do it in class 
design = MS(allvars)
X = design.fit_transform(stocks)
y = stocks.Direction == 'Up'
glm = sm.GLM(y,
             X,
             family=sm.families.Binomial())
results = glm.fit()
print(results.summary())

# This is the Scikit way 
model = LogisticRegression(random_state=42).fit(X, y)
print(model.score(X, y))
print(model.coef_)