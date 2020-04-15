import pandas as pd
import os
from dfply import * # https://github.com/kieferk/dfply#rename
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#set working directory

#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
    # The LDZ represents gas consumption in GWh, Actual is the Actual temperature and Normal is the normal temperature
conso = pd.read_csv(r'C:\Users\Nathalie\Documents\ENSTA\IN104\Projet\DE.csv',sep = ";")
conso.head()

    # Try to use dfply pipes to rename
#conso >> mutate(Date = pd.to_datetime(conso['Date (CET)']))


    # Plot using Matplotlib all three series on 3 sub plots to see them varying together
    # Do not forget to add a legend and a title to the plot

# plt.plot(conso['LDZ'],'r')
# plt.plot(conso['Actual'],'b')
# plt.plot(conso['Normal'],'y')
# plt.show()
# fig = plt.figure()
#
# fig.add_subplot(221)   #top left
# plt.plot(conso['LDZ'],'r')
# plt.xlabel("Serie 1:Consumption")
# fig.add_subplot(222)   #top right
# plt.plot(conso['Actual'],'b')
# plt.xlabel("Serie 2:Actual Temperature")
# fig.add_subplot(223)   #bottom left
# plt.plot(conso['Normal'],'y')
# plt.xlabel("Serie 3:Normal Temperature")
# plt.show()


    # Comment on their variation and their relationships

    # use dfply to transform Date column to DateTime type
conso['Date']= pd.to_datetime(conso['Date (CET)'])


#2) work on consumption data (non-linear regression)
#2)1. Plot with a scatter plot the consumption as a function of temperature
# plt.scatter(conso['Actual'],conso['LDZ'])
# plt.title('the consumption as a function of temperature')
# plt.xlabel('Temperature')
# plt.ylabel('Consumption')
# plt.show()

#2)2. define the consumption function (I give it to you since it is hard to know it without experience)

#This function takes temperature and 4 other curve shape parameters and returns a value of consumption
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#These are random initial values of a, b, c, d
guess_values= [500, -25, 2, 100]

#2)3. Fill out this h_hat array with values from the function h

# You will take the 'Actual' column from the DE.csv file as being the input temperature so its length should be the number of rows in the DataFrame imported
h_hat = np.empty(len(conso['LDZ']))

for i in range(len(conso['Actual'])):
    h_hat[i]=h(conso['Actual'][i],guess_values[0],guess_values[1],guess_values[2],guess_values[3])

    # For each value of temmperature of this column you will calculate the consumption using the h function above
    # Use the array guess_values for the curve parameters a, b, c, d that is to say a = guess_values[0], b = guess_values[1], c = guess_values[2], d = guess_values[3]
# plt.plot(conso['Actual'],h_hat,'r')
# plt.title('the h_hat values as a function of Actual temperature')
# plt.xlabel('Actual Temperature')
# plt.ylabel('h_hat')
# plt.plot(conso['Actual'],conso['LDZ'])
# plt.show()
    # Plot on a graph the real consumption (LDZ column) as a function of Actual temperature use blue dots
    # On this same graph add the h_hat values as a function of Actual temperature use a red line for this
    # Do not forget to add a legend and a title to the plot
    # Play around with the parameters in guess_values until you feel like your curve is more or less correct


#2)4. optimize the parameters

    # Your goal right now is to find the optimal values of a, b, c, d using SciPy
    # Inspire yourselves from the following video
    # https://www.youtube.com/watch?v=4vryPwLtjIY

#2)5. check the new fit

T=conso['Actual']
real_consumption=conso['LDZ']
T = np.nan_to_num(T)
real_consumption = np.nan_to_num(real_consumption)
c,cov=curve_fit(h,T,real_consumption,guess_values)
h_reg = np.empty(len(real_consumption))
for i in range(len(T)):
    h_reg[i]=h(T[i],c[0],c[1],c[2],c[3])
plt.plot(T,h_reg,'r')
#plt.title('the h_hat values as a function of Actual temperature')
#plt.xlabel('Actual Temperature')
#plt.ylabel('h_hat')
plt.scatter(T,real_consumption)
plt.show()

print('R^2:',r2_score(real_consumption,h_reg))


#Repeat what we did in 2)3. but with the new optimized coefficients a, b, c, d


#calculate goodness of fit parameters: correlation, root mean square error (RMSE), Average normalised RMSE, normalized RMSE
#averaged normalized RMSE is RMSE/(average value of real consumption)
#normalized RMSE is RMSE/(max value of real consumption - min value of real consumption)
#Any other metric we could use ?


