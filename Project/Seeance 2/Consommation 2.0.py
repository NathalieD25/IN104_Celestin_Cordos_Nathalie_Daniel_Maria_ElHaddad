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
from sklearn import metrics
import numpy as np
pd.options.mode.chained_assignment = None
import sklearn.metrics as metrics
##Random forest moduls 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import itertools

def split_fs(f, p):
    f = f >> mutate(NW = -X.injection + X.withdrawal)
    f['FSW1'] = f.apply (create_fsw1, axis=1)
    f['FSW2'] = f.apply (create_fsw2, axis=1)
    f['FSI1'] = f.apply (create_fsi1, axis=1)
    f['FSI2'] = f.apply (create_fsi2, axis=1)
    f['NW_b'] = f.apply (create_binary_nw, axis=1)
    f = (f >> inner_join(p, by='gasDayStartedOn') >> drop(0, 2, 3, 4, 5, 6, 7, 8, 9, 10)).dropna()
    return f

#fonction initialisation
seuil =45
def initialisation_data (s = 'storage_data.xlsx', p = 'price_data.csv'):
    storage_data = pd.read_excel(s, sheet_name=None)
    price_data = pd.read_csv(p)
    price_data.rename(columns={'Date':'gasDayStartedOn'}, inplace=True)
    price_data['gasDayStartedOn'] = pd.to_datetime(price_data['gasDayStartedOn'])
    storage_data = {k: split_fs(v, price_data) for k, v in storage_data.items()}
    return storage_data
    

    #creating empty dictionnaries
    #creating empty dictionnaries
    #for the logistic regression
    model1={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  
    #for the random forest 
    model2={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  
    dict_regression = dict () #model 3
    


###################### LOGISTIC REGRESSION #######################
def Logistic_Regression(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    print(lr.intercept_)
    y_pred = lr.predict(x_test)
    cm=metrics.confusion_matrix(y_test, y_pred)
    lr.predict_proba(x_test)
    df = pd.DataFrame({'x': x_test[:,0], 'y': y_test})
    df = df.sort_values(by='x')
    from scipy.special import expit
    sigmoid_function = expit(df['x'] * lr.coef_[0][0] + lr.intercept_[0]).ravel()
    plt.plot(df['x'], sigmoid_function)
    plt.scatter(df['x'], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')
    d = {'recall': metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]),"confusion": cm,"precision": metrics.precision_score(y_test, y_pred), "neg_precision":cm[1,1]/cm.sum(axis=1)[1], "roc": metrics.roc_auc_score(y_test,y_pred),"class_mod":lr}
    return d 
####################END OF THE LOGISTIC REGRESSION #############################



##########RANDOM FOREST PROGRAM #######################
######RANDOM FOREST###############
def random_forest(x,y):
    RSEED = 50
    train, test, train_labels, test_labels = train_test_split(x,y,test_size = 0.2, random_state = RSEED)

# Imputation of missing values
    train = np.nan_to_num(train)
    test = np.nan_to_num(test)

# Features for feature importances
    #features = list(train.columns)

# Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
    model.fit(train, train_labels)


    n_nodes = []
    max_depths = []

# Stats about the trees in random forest
    for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)


# Training predictions (to demonstrate overfitting)
    train_rf_predictions = model.predict(train)
    train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
    rf_predictions = model.predict(test)
    rf_probs = model.predict_proba(test)[:, 1]

 

# Plot formatting
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18

    def evaluate_model(predictions, probs, train_predictions, train_probs):
    # """Compare machine learning model to baseline performance.
    # Computes statistics and shows ROC curve."""
    #
        baseline = {}

        baseline['recall'] = recall_score(test_labels,
                                     [1 for _ in range(len(test_labels))])
        baseline['precision'] = precision_score(test_labels,
                                      [1 for _ in range(len(test_labels))])
        baseline['roc'] = 0.5

        results = {}

        results['recall'] = recall_score(test_labels, predictions)
        results['precision'] = precision_score(test_labels, predictions)
        results['roc'] = roc_auc_score(test_labels, probs)

        train_results = {}
        train_results['recall'] = recall_score(train_labels, train_predictions)
        train_results['precision'] = precision_score(train_labels, train_predictions)
        train_results['roc'] = roc_auc_score(train_labels, train_probs)

        for metric in ['recall', 'precision', 'roc']:
                print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
        model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    cm = metrics.confusion_matrix(test_labels, rf_predictions)
    d2= {"recall": metrics.recall_score(test_labels, rf_predictions), 
         "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]), 
         "confusion": cm,
         "precision": metrics.precision_score(test_labels, rf_predictions), 
         "neg_precision":cm[1,1]/cm.sum(axis=1)[1], 
         "roc": metrics.roc_auc_score(test_labels, rf_predictions)} #ajouter le modele 
    return d2
###############END OF THE RANDOM FOREST PROGRAM ###################################




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
        x = np.array(dataFrame[feature_cols]) # Features
        y = np.array(dataFrame['Net Withdrawal binary']) # Target variable
        model1[k]=Logistic_Regression(x,y)
        model2[k]=random_forest(x,y)
        
        #ajouter fonction pour comprarer et choisir le meilleur
        
        
        
################REGRESSION###############
        X = dataFrame[feature_cols] # Features
        y = dataFrame['Net Withdrawal'] # Target variable

       
        
        
        #plt.figure(figsize=(15,10))
        #plt.tight_layout()
        #seabornInstance.distplot(dataset['quality'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        regressor = LinearRegression()  
        l_reg = regressor.fit(X_train, y_train)
        
        coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
        coeff_df
        
        y_pred = regressor.predict(X_test)
        
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df1 = df.head(25)
        
        df1.plot(kind='bar',figsize=(10,8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        
        #np.mean ()
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        
        n = len (y_test)
        averageValueConsumption = 0
        minValue = min (y_test)
        for i in range (n):
            averageValueConsumption += y_test.values[i]
            
        
        averageValueConsumption /= n
        maxValueConsumption = np.max (y_test)
        ANRMSE = RMSE/averageValueConsumption
        NRMSE = RMSE/(maxValueConsumption - minValue)
        r2 = metrics.r2_score(y_test, y_pred)
        d_regression = {'r2': r2, 'rmse': RMSE, 'nrmse': NRMSE, 'anrmse': ANRMSE, 'l_reg':l_reg }#pas complet des choses à comprendre et à completer
        dict_regression[k] = d_regression
        
        


