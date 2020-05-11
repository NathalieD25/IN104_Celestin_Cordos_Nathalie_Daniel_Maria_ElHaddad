# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:32:48 2020

@author: Celestin Cordos
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#matplotlib inline


storage_data = pd.read_excel("storage_data.xlsx", sheet_name=None)
price_data = pd.read_csv('price_data.csv', sep = ';')
price_data.rename(columns={'Date':'gasDayStartedOn '}, inplace=True)


dataset = pd.read_csv('winequality.csv')

dataset.shape 
dataset.describe ()

dataset.isnull().any()

dataset = dataset.fillna(method='ffill')

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']]
y = dataset['quality']


#plt.figure(figsize=(15,10))
#plt.tight_layout()
#seabornInstance.distplot(dataset['quality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

n = len (y_test)
averageValueConsumption = 0
minValue = min (y_test)
for i in range (n):
    averageValueConsumption += y_test.values[i]
    

averageValueConsumption /= n

ANRMSE = RMSE/averageValueConsumption
NRMSE = RMSE/(averageValueConsumption - minValue)
r2 = metrics.r2_score(y_test, y_pred)