import pandas as pd

df = pd.read_csv('pv.csv')

x = df[['irr', 'tamb', 'tmodule', 'wind']].values
y = df[['pack']].values

print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30) 
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print(reg.coef_, reg.intercept_)

y_hat = reg.predict(x_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(y_test, y_hat)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test, y_hat)
print(f'R Squared score     :{r2}')

reg_1 = linear_model.Lasso()
reg_1.fit(x_train, y_train)
print(reg_1.coef_, reg_1.intercept_)

y_hat = reg_1.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test, y_hat)
print(f'R Squared score     :{r2}')

###