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

Auto = load_data('Auto') # The course data is at https://www.statlearning.com/resources-python 
print(Auto.describe())

Auto.drop('name', inplace=True, axis=1)
cols = Auto.shape[1]
pos = 1
dots = 5
fig = plt.figure(figsize=(9,9))

for columnA in Auto:
    for columnB in Auto:
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
            ax.text(0.5, 0.5, Auto[columnA].name, fontsize=12, ha='center', va='center')
        else:
            ax.scatter(Auto[columnB], Auto[columnA], s=dots)
            # Organized so that each label is X for plots in its column 
        pos +=1
plt.tight_layout()            
y = Auto['mpg']
predictors = Auto.columns.drop(['mpg', 'horsepower']) # Less typing this way & it adds intercept automatically 
design = MS([poly('weight', degree=2), 'year', 'origin', 'acceleration'])
X = design.fit_transform(Auto)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# table = anova_lm(results)
table = anova_lm(*[sm.OLS(y, D).fit() for D in design.build_sequence(Auto, anova_type='sequential')])
table.index = design.names
print(table)

fig2 = plt.figure(figsize=(9,5))
axx1 = fig2.add_subplot(1,2,1)
axx1.scatter(results.fittedvalues, results.resid_pearson)
axx1.set_xlabel('Fitted Values')
axx1.set_ylabel('Residuals')
axx1.axhline(0, c='red', linewidth=2)
axx2 = fig2.add_subplot(1,2,2)
axx2 = sm.graphics.plot_leverage_resid2(results, ax = axx2)
plt.show()