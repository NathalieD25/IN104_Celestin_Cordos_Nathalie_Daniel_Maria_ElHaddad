# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:13:49 2020

@author: Celestin Cordos
"""

import pandas as pd
import dfply 

from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np



seuil =45

storage_data = pd.read_excel("storage_data.xlsx", sheet_name=None)
price_data = pd.read_csv('price_data.csv')
price_data.rename(columns={'Date':'gasDayStartedOn '}, inplace=True)

lr = LogisticRegression()

for k, v in storage_data.items():
    if k != 'SF - UGS Stassfurt':
        dataFrame = storage_data [k]
        dataFrame.dropna()
        
        del dataFrame ['status'] 
        del dataFrame ['trend']
        n = dataFrame.shape [0]
        
        netWithdrawal = [0]*n
        FSW1 = [0]*n
        FSW2 = [0]*n
        netWithdrawalBinary = [0]*n
        netWithdrawalLagged = [0]* n
        
        for i in range (n):
            netWithdrawal [i] = dataFrame['withdrawal'][i] - dataFrame['injection'][i]
            FSW1[i] = max (dataFrame['full'][i] - seuil, 0)
            FSW2[i] = max (- dataFrame['full'][i] + seuil, 0)
            if netWithdrawal[i] > 0:
                netWithdrawalBinary [i] = 1
            if i >  0:
                netWithdrawalLagged [i -1 ] = netWithdrawal [i]
        
        
        
        
        netWithdrawal [i] = 0
        FSW1[i] = max (dataFrame['full'][i] - seuil, 0)
        FSW2[i] = max (- dataFrame['full'][i] + seuil, 0)
        
        dataFrame ['Net Withdrawal binary'] = netWithdrawalBinary
        dataFrame ['Net Withdrawal'] = netWithdrawal
        dataFrame ['Net Withdrawal Lagged'] = netWithdrawalLagged
        dataFrame ['FSW1'] = FSW1
        dataFrame ['FSW2'] = FSW2
        
        dataFrame.dropna (inplace = True )
        
        storage_data[k] = dataFrame
    #    dfply.join.inner_join(dataFrame, price_data)

#deuxieme partie a partir d'ici
        feature_cols = ['Net Withdrawal Lagged', 'FSW1', 'FSW2']
        X = dataFrame[feature_cols] # Features
        y = dataFrame['Net Withdrawal binary'] # Target variable
        
        
        lr.fit(X, y)
        
        print(lr.coef_)
        print(lr.intercept_)
        
        
X_test = 
#y_pred = lr.predict(x_test)
#
#confusion_matrix(y_test, y_pred)
#
#
#df = pd.DataFrame({'x': x_test[:,0], 'y': y_test})
#df = df.sort_values(by='x')
#from scipy.special import expit
#sigmoid_function = expit(df['x'] * lr.coef_[0][0] + lr.intercept_[0]).ravel()
#plt.plot(df['x'], sigmoid_function)
#plt.scatter(df['x'], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')


