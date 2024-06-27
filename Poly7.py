###
### Demonstrating bias-variance with a simulated set of known data
###
# I can NOT get it to overfit using polynomials or splines - not enough noise in the simul data? 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
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

Data = pd.read_csv(os.getcwd() + '/Quartic.csv')   
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
    bspline = MS([bs('X', df=d, degree=3)]).fit(train_set) # Using this ISLP helper function does not seem like a great idea
    # spline_model = smf.ols(formula='y ~ bs(x, df=d, degree=3, include_intercept=True)', data=train_set).fit() # Normal smf idiom
    X_train = bspline.transform(train_set)
    X_test = bspline.transform(test_set)
    model = sm.OLS(y_train, X_train).fit()
    train_mse.append(mse(y_train, X_train, model))
    test_mse.append(mse(y_test, X_test, model))

fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(df, test_mse, label='Test')
ax1.set_xlabel('DF')
ax1.set_ylabel('MSE')
ax1.plot(df, train_mse, label='Train')
ax1.legend();

best_test = np.column_stack((df, test_mse))
min_at = np.argmin(best_test[:, 1])
best_df = int(best_test[min_at, 0])
print(best_test[min_at, 1])
print(best_df)

plt.show()

n_knots = best_df - 3 # Apparently not -4 going from SM to Sklearn 
# Here, it was 6 df to make 3 knots but only 1 in the middle 
spline_transformer = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
model2 = make_pipeline(spline_transformer, LinearRegression())
X = np.asarray(Data['X']).reshape((-1,1)) # Something about needing to be a matrix, p. 318 
Y = np.asarray(Data['Y']).reshape((-1,1))
model2.fit(X, Y) # This is post-testing, so use all data 

print(mse(Y, X, model2))

knots = spline_transformer.bsplines_[0].t
internal_knots = knots[4:-4]

fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(1,1,1)
y_pred = model2.predict(X)
ax2.plot(X, y_pred, color='red')
ax2.scatter(X, Y, s=9, alpha=0.6, color='grey')
for knot in internal_knots:
    plt.axvline(knot, color='blue', linestyle='--', alpha=0.7)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.show()