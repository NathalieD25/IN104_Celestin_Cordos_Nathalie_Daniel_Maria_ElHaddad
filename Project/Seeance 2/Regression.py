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

seuil =45 #pour FSW. Fixe normalement mais si jamais on veut le changer...
dict_regression = dict ()


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
        y = dataFrame['Net Withdrawal'] # Target variable

       
        
        
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
        d_regression = {'r2': r2, 'rmse': RMSE, 'nrmse': NRMSE, 'anrmse': ANRMSE}#pas complet des choses à comprendre et à completer
        dict_regression[k] = d_regression