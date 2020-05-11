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

import itertools

seuil =45

storage_data = pd.read_excel("storage_data.xlsx", sheet_name=None)
price_data = pd.read_csv('price_data.csv')
price_data.rename(columns={'Date':'gasDayStartedOn '}, inplace=True)

#creating empty dictionnaries
#creating empty dictionnaries
#for the logistic regression
model1={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  
#for the random forest 
model2={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  


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
    train, test, train_labels, test_labels = train_test_split(x,y,test_size = 0.3, random_state = RSEED)

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
    cm = confusion_matrix(test_labels, rf_predictions)
    d2= {"recall": metrics.recall_score(test_labels, rf_predictions), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]), "confusion": cm,"precision": metrics.precision_score(test_labels, rf_predictions), "neg_precision":cm[1,1]/cm.sum(axis=1)[1], "roc": metrics.roc_auc_score(test_labels, rf_predictions)}
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
        
        


