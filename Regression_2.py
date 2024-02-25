import pandas as pd

df = pd.read_csv('pv.csv')

x = df[['irr', 'tamb', 'tmodule', 'wind']].values
y = df[['pack']].values

print(x.shape, y.shape)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold
kf = KFold(n_splits=4)

for train_index, test_index in kf.split(x):
    print()
    #Train-Test Split
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    # Inference
    y_predict = reg.predict(x_test)
    
    #Evaluation
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print(f'Mean Squared Error  : {mse}')
    print(f'R Squared score     :{r2}')
