import dfply
import supply
import DEMAND
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy



import os
from openpyxl.workbook import Workbook
def market_decision(DF):
    decision=[]
    real_decision=[]
    for i in range (0,len(DF)):
        if DF['Supply'][i]>DF['Demand'][i]:
            decision.append("SELL")

            #print ("On %s : decision is SELL" %(DF['Date'][i]))
        if DF['Supply'][i]<DF['Demand'][i]:
            #print ("On %s : decision is BUY" %(DF['Date'][i]))
            decision.append("BUY")
        if DF['Supply'][i]==DF['Demand'][i]:
            #print ("On %s : decision is FLAT" %(DF['Date'][i]))
            decision.append("FLAT")
        if DF['Supply_real'][i]>DF['Demand_real'][i]:
            real_decision.append("SELL")
        if DF['Supply_real'][i]<DF['Demand_real'][i]:
            real_decision.append("BUY")
        if DF['Supply_real'][i]==DF['Demand_real'][i]:
            real_decision.append("FLAT")

    return decision,real_decision

def metrics (balance):
    RMSE_d = np.sqrt(mean_squared_error(balance['Demand_real'], balance['Demand'])) 
    averageValueConsumption_d = np.mean (balance['Demand_real'])
    maxValueConsumption_d = np.max (balance['Demand_real'])
    minValueConsumption_d = np.min (balance['Demand_real'])
    ANRMSE = RMSE_d/averageValueConsumption_d
    NRMSE = RMSE_d/(maxValueConsumption_d - minValueConsumption_d)
    corr = scipy.stats.pearsonr(balance['Demand_real'],balance['Demand'])[0]
    d_demand = {'rmse': RMSE_d, 'nrmse': NRMSE, 'anrmse': ANRMSE, 'corr': corr }
    
    RMSE_s = np.sqrt(mean_squared_error(balance['Supply_real'], balance['Supply'])) 
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
    balance["Decision"]=market_decision(balance)[0]
    balance["Decision_real"]=market_decision(balance)[1]
    balance = balance[(balance.T != 0).all()]
#    balance.replace(0, np.nan, inplace = True)
#    balance.dropna(how='all', axis=0, inplace = True)
#    balance.replace(np.nan, 0, inplace = True)
    
    
    balance.to_csv('final_balance.csv',index=True)
    d_demand, d_supply = metrics (balance)
    print(balance)
    print (d_demand)
    print (d_supply)
    
    
   
    
    
    


if __name__ == '__main__':
    main ()
