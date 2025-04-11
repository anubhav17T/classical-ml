from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 2, 3],
              [1, 4, 5],
              [1, 7, 8]])
y = np.array([[6],
              [9],
              [15]])

lr = LinearRegression()
lr.fit(X=X, y=y)
print(lr.coef_)
print(lr.intercept_)
