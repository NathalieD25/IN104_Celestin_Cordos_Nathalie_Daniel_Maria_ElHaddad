
import dfply
import DEMAND
import supply
import os
from openpyxl.workbook import Workbook
def market_decision(DF):
    decision=[]
    for i in range (0,len(balance)):
        if DF['Supply'][i]>DF['Demand'][i]:
            decision.append("SELL")

            #print ("On %s : decision is SELL" %(DF['Date'][i]))
        if DF['Supply'][i]<DF['Demand'][i]:
            #print ("On %s : decision is BUY" %(DF['Date'][i]))
            decision.append("BUY")
        if DF['Supply'][i]==DF['Demand'][i]:
            #print ("On %s : decision is FLAT" %(DF['Date'][i]))
            decision.append("FLAT")

    return decision

def main():
    DF_demand=df
    DF_supply=Supply
    DF_supply.rename(columns={"gasDayStartedOn":"Date"},inplace=True)
    DF_demand['Date'] = pd.to_datetime(DF_demand['Date'])
    DF_supply['Date'] = pd.to_datetime(DF_supply['Date'])

    balance=DF_demand>> inner_join(DF_supply,by='Date') ##inner joining the 2 date frames
    c=market_decision(balance)
    balance["Decision"]=market_decision(balance)
    balance.to_excel("Balance.xlsx")
    print(balance)
