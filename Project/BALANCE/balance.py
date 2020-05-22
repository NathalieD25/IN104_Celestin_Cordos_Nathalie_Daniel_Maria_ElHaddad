import dfply
import supply
import DEMAND
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy


from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score




import os
from openpyxl.workbook import Workbook
def market_decision(DF):
    decision=[]
    real_decision=[]
    DF['Decision'] = 0
    DF['Decision_real'] = 0
    for i in range (0,len(DF)):
        if DF['Supply'][i]>DF['Demand'][i]:
            DF['Decision'][i] = 'SELL'

            #print ("On %s : decision is SELL" %(DF['Date'][i]))
        if DF['Supply'][i]<DF['Demand'][i]:
            #print ("On %s : decision is BUY" %(DF['Date'][i]))
            DF['Decision'][i] ='BUY'
        if DF['Supply'][i]==DF['Demand'][i]:
            #print ("On %s : decision is FLAT" %(DF['Date'][i]))
            DF['Decision'][i] ='FLAT'
        if DF['Supply_real'][i]>DF['Demand_real'][i]:
            DF['Decision_real'][i] ='SELL'
        if DF['Supply_real'][i]<DF['Demand_real'][i]:
            DF['Decision_real'][i]='BUY'
        if DF['Supply_real'][i]==DF['Demand_real'][i]:
            DF['Decision_real'][i] = 'FLAT'
        


def metrics (balance):
    RMSE_d = mean_squared_error(balance['Demand_real'], balance['Demand'])
    averageValueConsumption_d = np.mean (balance['Demand_real'])
    maxValueConsumption_d = np.max (balance['Demand_real'])
    minValueConsumption_d = np.min (balance['Demand_real'])
    ANRMSE = RMSE_d/averageValueConsumption_d
    NRMSE = RMSE_d/(maxValueConsumption_d - minValueConsumption_d)
    corr = scipy.stats.pearsonr(balance['Demand_real'],balance['Demand'])[0]
    d_demand = {'rmse': RMSE_d, 'nrmse': NRMSE, 'anrmse': ANRMSE, 'corr': corr }
    
    RMSE_s = mean_squared_error(balance['Supply_real'], balance['Supply']) 
    averageValueConsumption_s = np.mean (balance['Supply_real'])
    maxValueConsumption_s = np.max (balance['Supply_real'])
    minValueConsumption_s = np.min (balance['Supply_real'])
    ANRMSE = RMSE_s/averageValueConsumption_s
    NRMSE = RMSE_s/(maxValueConsumption_s - minValueConsumption_s)
    corr = scipy.stats.pearsonr(balance['Supply_real'],balance['Supply'])[0]
    d_supply = {'rmse': RMSE_d, 'nrmse': NRMSE, 'anrmse': ANRMSE, 'corr': corr }
    
    
    
    return d_demand, d_supply
    

def main():
    DF_demand=DEMAND.df
    DF_supply=supply.Supply
    DF_supply.rename(columns={"gasDayStartedOn":"Date"},inplace=True)
    DF_demand['Date'] = pd.to_datetime(DF_demand['Date'])
    DF_supply['Date'] = pd.to_datetime(DF_supply['Date'])

    balance=DF_demand>> dfply.inner_join(DF_supply,by='Date') ##inner joining the 2 date frames
    market_decision(balance)

    balance = balance[(balance.T != 0).all()]

    
    
    balance.to_csv('final_balance.csv',index=True)
    d_demand, d_supply = metrics (balance)
    print(balance)
    print ('demand metrics : ', d_demand)
    print ('supply metrics : ',  d_supply)
    for i in range (len (balance)):
        if balance ['Decision'].values[i] == 'SELL':
            balance['Decision'].values[i] = 1
        if balance ['Decision'].values[i] == 'BUY':
            balance['Decision'].values[i] = 0
        if balance ['Decision'].values[i] == 'FLAT':
            balance['Decision'].values[i] = 2
        if balance ['Decision_real'].values[i] == 'SELL':
            balance['Decision_real'].values[i] = 1
        if balance ['Decision_real'].values[i] == 'BUY':
            balance['Decision_real'].values[i] = 0
        if balance ['Decision_real'].values[i] == 'FLAT':
            balance['Decision_real'].values[i] = 2
            
    y_test, y_pred = list(balance['Decision_real']),  list(balance['Decision'])    
    cm = confusion_matrix (y_test, y_pred)
    d_decision = {'recall': recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]),"confusion": cm,"precision": precision_score(y_test, y_pred), "neg_precision":cm[1,1]/cm.sum(axis=1)[1], "roc": roc_auc_score(y_test,y_pred)}
    print ('decision metrics : ', d_decision)
    
   
    
    
    


if __name__ == '__main__':
    main ()
