import dfply
import supply
import DEMAND
import pandas as pd


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

def main():
    DF_demand=DEMAND.df
    DF_supply=supply.Supply
    DF_supply.rename(columns={"gasDayStartedOn":"Date"},inplace=True)
    DF_demand['Date'] = pd.to_datetime(DF_demand['Date'])
    DF_supply['Date'] = pd.to_datetime(DF_supply['Date'])

    balance=DF_demand>> dfply.inner_join(DF_supply,by='Date') ##inner joining the 2 date frames
    balance["Decision"]=market_decision(balance)[0]
    balance["Decision_real"]=market_decision(balance)[1]
    balance=balance[(balance.T!=0).any()]
    #balance.to_csv('final_balance.csv',index=True)
   
    
    
    print(balance)


if __name__ == '__main__':
    main ()
