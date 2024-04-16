# The labs for the ISL course are notebooks at https://github.com/intro-stat-learning/ISLP_labs 
# So I am doing them on Google Colab 
# The exercises, I am doing in Studio 

from tracemalloc import stop
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
y_pred = results.get_prediction(X)

def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
    
ax = Auto.plot.scatter('horsepower', 'mpg')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)

plt.show()