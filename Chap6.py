###
### I want to plot the Ridge Regression with varying Lambda like in the book
### 
from ISLP.models.columns import OrdinalEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_object_dtype
import seaborn as sns
sns.set_theme() 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
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

Data = pd.read_csv('F:\\Studio\\ISL_Course\\Datasets\\Credit.csv') 
frame_stats(Data)  
cat_vars = ['Own', 'Student', 'Married', 'Region']
num_vars = ['Income', 'Limit', 'Cards', 'Age', 'Education', 'Rating'] 
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy = 'median')),
    ('scaler', StandardScaler())
    ])
full_pipeline = ColumnTransformer(
    [
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-999), cat_vars),
        #('binned', KBinsDiscretizer(n_bins=10), ['Income', 'Balance']), # Will binning work with standardization? 
        ('num', num_pipeline, num_vars),
    ],
    remainder="drop",
)

coeffs = []
lambdas = []
X = full_pipeline.fit_transform(Data) 
y = Data['Balance']
for l in np.logspace(-2, 6, base=10):
    model = Ridge(alpha = l)
    model.fit(X, y)
    coeffs.append(model.coef_ )
    lambdas.append(l)

coefficients = pd.DataFrame(coeffs, columns=cat_vars+num_vars)
print(coefficients)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1,1,1)
#ax.plot(coeffs)
ax.set_ylabel('Coefficients')
ax.set_xlabel('Lambda')
ax.semilogx(lambdas, coeffs)
plt.show()
