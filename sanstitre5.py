from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

data=datasets.load_boston()
X=data.data[:,-1]
y=data.target
X,y=X.reshape(-1,1), y.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
regr = linear_model.LinearRegression() #instanciation du modèle
regr.fit(X_train, y_train)
print(regr.coef_, regr.intercept_)
plt.scatter(X_train,y_train)