import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

#This function sets the working directory
def set_wd(wd):
    os.chdir(wd)



#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimeter = ";", plot = True):
    myData = pd.read_csv('DE.csv', sep = ';')
    # Try to use dfply pipes to rename
    myData.rename(columns={"Date (CET)": "Date"},inplace=True)

    if plot:
        fig, (ax1, ax2,ax3) = plt.subplots(3)
        fig.suptitle('Horizontally stacked subplots of the three series of data')
        ax1.plot(myData['LDZ'],'r')
        ax1.set_title('LDZ: Gas consumption in GWh')
        ax2.plot(myData['Actual'],'b')
        ax2.set_title('Actual temperatures in °C')
        ax3.plot(myData['Normal'],'g')
        ax3.set_title('Normal temperatures in °C')
        plt.show()
        plt.close()
    ##Removing the Nans from our data frames using the nan_to_num function
    myData['Actual']=np.nan_to_num(myData['Actual'])
    myData['LDZ']=np.nan_to_num(myData['LDZ'])
    myData['Normal']=np.nan_to_num(myData['Normal'])
    myData['Date']=np.nan_to_num(myData['Date'])

    return myData >> mutate(Date = pd.to_datetime(myData['Date']))

#This function creates a scatter plot given a DataFrame and an x and y column


def scatter_plot(dataframe="conso" , x = "Actual", y = "LDZ", col = "red"):
    X,Y = list(conso[x]), list(conso[y])
    plt.scatter (X, Y, c = col)
    plt.title('The consumption as a function of temperature')
    plt.xlabel('Actual tempartures (in °C)')
    plt.ylabel('Gas consumption (in GWh)')
    plt.show ()
    plt.close()

#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(t, real_conso, a = 900, b = -35, c = 6, d = 300, plot = True):
    h_hat = np.empty(len(t))
    for i in range(len(t)):
       h_hat[i] = h(t[i], a, b, c, d)
    if plot:
        plt.plot(t, h_hat, c= 'yellow',label='Gas consumption using the sigmoid function as a function of Actual temperature')
        plt.xlabel('Actual tempartures (in °C)')
        plt.ylabel('Gas consumption (in GWh)')
        #if real_conso is not None you plot it as well
        if not isinstance(real_conso, type(None)):
            plt.scatter(t, real_conso, c = 'green',label='The real gas consumption as given in the data frame as a function of Actual temperature')
            plt.xlabel('Actual tempartures (in °C)')
            plt.ylabel('Gas consumption (in GWh)')
            plt.legend(loc="upper right")

            if(len(t) != len(real_conso)):
                print("Difference in length between Temperature and Real Consumption vectors")
            plt.show ()
            plt.close()

        #legend((line1, line2), ('label1', 'label2', 'label3'))
    return h_hat

#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
        LDZ = real_conso #so we can use our parameter previously defined
        RMSE = 0.0
        averageValueConsumption = 0
        minValue = min (LDZ)
        n = len (LDZ)
        for i in range (n):
            RMSE+= (LDZ[i]-h_hat[i])**2
            averageValueConsumption += LDZ[i]


        RMSE = np.sqrt (RMSE/n)
        averageValueConsumption /= n
        R2 = r2_score(h_hat, LDZ)


        print ('R^2 = ', R2)
        print ('RMSE : ', RMSE)
        print ('normalized RMSE : ', RMSE/(averageValueConsumption - minValue))
        print ('Average normalised RMSE : ', RMSE/averageValueConsumption)

    return [R2 ,RMSE,   RMSE/(averageValueConsumption - minValue), RMSE/averageValueConsumption]# returns [RMSE,Average  RMSE, normalized RMSEnormalised]

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        cons = []
        for i in range (len ((temperature))):
            cons.append (h(temperature [i], self.a,self.b,self.c,self.d))
        return cons


    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):
        temperature = [i for i in range (-40,40,1)]
        real_conso = self.get_consumption (temperature)
        consumption_sigmoid(temperature,None,  self.a,  self.b,  self.c,  self.d, p)

    #This is what the class print if you use the print function on it
    def __str__(self):
        print ('\n'"a : ", self.a ,"b : ",self.b , "c : ",self.c ,"d : ",self.d  )
        t = "Print Consumption"
        return t

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    __guess_a, __guess_b, __guess_c, __guess_d = 900,  -35,  6, 300

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
#                self.Actual = f['Actual']
#                self.LDZ = f['LDZ'] # not really necessary
                self.__f = f
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.__coef, self.__cov = curve_fit(
                h,
                list (self.__f['Actual']),
                list (self.__f['LDZ']),
                [self.__guess_a, self.__guess_b, self.__guess_c, self.__guess_d]
                )

            s = consumption_sigmoid(list (self.__f['Actual']), list (self.__f['LDZ']), a = self.__coef[0], b = self.__coef[1], c = self.__coef[2], d = self.__coef[3], plot = True         )
            print ("s", s)
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, self.__f['LDZ'])
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if self.__corr is not None: # any other should do
            return [self.__corr, self.__rmse, self.__nrmse, self.__anrmse]
        else:
            print("optimize method is not yet run")

    #This function creates the class consumption
    def create_consumption(self):
        if self.__corr is not None: #same here
            return consumption (self.__coef[0], self.__coef[1], self.__coef[2], self.__coef[3])
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        t = "Print optimize sigmoid"
        if self.__corr is not None:

            print ('\n', "a : ", self.__coef[0], "b : ", self.__coef[1],"a : ", self.__coef[2],"a : ", self.__coef[3])
            print ("R**2 : ",self.__corr, "RMSE : ",self.__rmse,"Normalized RMSE : ",self.__nrmse,"Normalized Average RMSE : ",self.__anrmse)

        else:
            t = "optimize method is not yet run"
        return t

#If you have filled correctly the following code will run without an issue
if __name__ == '__main__':

    #set working directory
    set_wd("Data")

    #1) import consumption data and plot it
    conso = import_csv()
    plt.close()
    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature

    scatter_plot(conso)

    plt.close()
    #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    sig.optimize()
    c = sig.create_consumption()
    print(sig)
    plt.close()

    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change

    print(
            [
            sig.__dict__['_optimize_sigmoid__corr'],
            sig.__dict__['_optimize_sigmoid__rmse'],
            sig.__dict__['_optimize_sigmoid__nrmse'],
            sig.__dict__['_optimize_sigmoid__anrmse']
            ]
        )

    print(
            [
            sig._optimize_sigmoid__corr,
            sig._optimize_sigmoid__rmse,
            sig._optimize_sigmoid__nrmse,
            sig._optimize_sigmoid__anrmse
            ]
        )

    print(
            [
            getattr(sig, "_optimize_sigmoid__corr"),
            getattr(sig, "_optimize_sigmoid__rmse"),
            getattr(sig, "_optimize_sigmoid__nrmse"),
            getattr(sig, "_optimize_sigmoid__anrmse")
            ]
        )

    print(sig.fit_metrics())
    c.sigmoid(True)
    print(c)

    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N day