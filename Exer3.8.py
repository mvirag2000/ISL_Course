# The labs for the ISL course are notebooks at https://github.com/intro-stat-learning/ISLP_labs 
# So I am doing them on Google Colab 
# The exercises, I am doing in Studio 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly) # This is the special ISLP library 

Auto = load_data('Auto') # The course data is at https://www.statlearning.com/resources-python 
print(Auto.describe())

X = pd.DataFrame({'intercept' : np.ones(Auto.shape[0]), # For OLS you have to seed this intercept vector 
                  'horsepower' : Auto['horsepower']})
y = Auto['mpg']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print(results.params)
print(summarize(results))
dots = 10
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(2,2,1)
ax1.scatter(Auto['horsepower'], Auto['mpg'], s=dots)
ax1.set_title('MPG due to HP')
abline(ax1,
       results.params[0],
       results.params[1],
       'red',
       linewidth=2)

design = MS(['horsepower']) # This is ModelSpec from the ISLP library
design = design.fit(Auto)
X = design.transform(Auto) # It handles multiple predictors, and the intercept 
Auto['resid'] = results.resid # statsmodels.regression.linear_model.OLSResults has many cool attributes
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('Residuals')
ax2.scatter(Auto['horsepower'], Auto['resid'], s=dots)

X['hp_squared'] = X['horsepower'] ** 2
model2 = sm.OLS(y, X)
results = model2.fit()
print(results.summary())
Auto['fitter'] = results.fittedvalues
Auto.sort_values('horsepower', inplace=True)
ax3 = fig.add_subplot(2,2,3)
ax3.set_title('Quadratic Model')
ax3.scatter(Auto['horsepower'], Auto['mpg'], s=dots)
ax3 = ax3.plot(Auto['horsepower'], Auto['fitter'], 'red', linewidth=2)

ax4 = fig.add_subplot(2,2,4)
ax4.set_title('Residuals')
Auto['resid'] = results.resid # Overwrite old residuals
ax4.scatter(Auto['horsepower'], Auto['resid'], s=dots)

plt.tight_layout()
plt.show()
