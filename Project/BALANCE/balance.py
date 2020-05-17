
import dfply
import DEMAND
#import supply

def market_decision(DF):

    decision=[]
    for i in range (0,len(DF)):
        if DF['Supply'][i]>DF['Demand'][i]:
            decision.append("SELL")

            #print ("On %s : decision is SELL" %(DF['Date'][i]))
        if DF['Supply'][i]<DF['Demand'][i]:
            #print ("On %s : decision is BUY" %(DF['Date'][i]))
            decision.append("BUY")
        else:
            #print ("On %s : decision is FLAT" %(DF['Date'][i]))
            decision.append("FLAT")

    return decision

def main():
    DF_demand=df

    DF_supply= pd.read_csv('Book1.csv', sep = ',') ####just to test the function: the DF_supply has to be a dataFRAME

    DF_demand['Date'] = pd.to_datetime(DF_demand['Date'])
    DF_supply['Date'] = pd.to_datetime(DF_supply['Date'])

    balance=DF_demand>> inner_join(DF_supply,by='Date') ##inner joining the 2 date frames

    balance["Decision"]=market_decision(balance)

    print(balance)
