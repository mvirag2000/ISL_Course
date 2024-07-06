##
##  I couldn't resist working the Hitters data with TF   
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'} 
from tensorflow import keras 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # Better for NN than StandardScalar 
from sklearn.impute import SimpleImputer
import seaborn as sns
sns.set_theme() 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# from sklearn.tree import DecisionTreeRegressor
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

Data = pd.read_csv(os.getcwd() + '/Hitters.csv').dropna()   
frame_stats(Data)

cat_vars = ['League', 'Division', 'NewLeague']
num_vars = ['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat', 'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks', 'PutOuts', 'Assists', 'Errors']

y = Data['Salary']

# Train the encoders
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', MinMaxScaler())
    ])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_vars),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars)
    ]) 

X = full_pipeline.fit_transform(Data) 
features = X.shape[1]

def build_model(n_hidden=1, n_neurons=256, learning_rate=0.001):
    print("\n*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(features,)))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1)) 
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    model.summary()
    return model

# Tip: If you suspect data problems, try a model that gives decent error messages
# model = DecisionTreeRegressor()
# model.fit(train_data_ready, train_y)

model = build_model(2, 256, 0.01) # Generic 2-layer NN

# Bump epochs out to 30 to see Double Descent from page 432 in the textbook 
history = model.fit(X, y, epochs=30, verbose=2, validation_split=0.2, shuffle=True, batch_size=32)

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Model MAE')
plt.legend(loc='lower left')
plt.ylabel('Mean Abs Error')
plt.xlabel('epoch')
plt.show()
