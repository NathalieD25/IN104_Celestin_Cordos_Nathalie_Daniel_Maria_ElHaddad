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
import dfply
import itertools
import scipy





def create_fsw1 (row):
    return  max (row['full'] - seuil, 0)

def create_fsw2 ( row):
    return max (- row['full'] + seuil, 0)

def create_binary_nw ( row):
    if row['NW'] > 0:
        return 1
    else:
        return 0
    



def split_fs( f, p):
    f = f >> dfply.mutate( NW = -f.injection + f.withdrawal)
    f['NW_Lagged'] = f['NW'].shift(1)
    f['FSW1'] = f.apply (create_fsw1, axis=1)
    f['FSW2'] = f.apply (create_fsw2, axis=1)
    
    ##pas compris à quoi ca servait; on les avait pas avant
    '''f['FSI1'] = f.apply (create_fsi1, axis=1)
    f['FSI2'] = f.apply (create_fsi2, axis=1)'''
    f['NW_b'] = f.apply (create_binary_nw, axis=1)
    f =  pd.merge(f,p, on='gasDayStartedOn', how= 'inner').dropna()
    return f


def initialisation_data ( s = 'storage_data.xlsx', p = 'price_data.csv'):
    storage_data = pd.read_excel(s, sheet_name=None)
    price_data = pd.read_csv(p, sep = ';')
    price_data.rename(columns={'Date':'gasDayStartedOn'}, inplace=True)
    price_data['gasDayStartedOn'] = pd.to_datetime(price_data['gasDayStartedOn'])
    storage_data = {k: split_fs(v, price_data) for k, v in storage_data.items()}
    return storage_data



class Classification ():
    def __init__ (self):
        self.Storages = storage_data
        self.model1={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  
        #for the random forest 
        self.model2={"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} }  
        self.coefficients = dict () #sera un dictionnaire de dictionnaire. le premier aura les cles 'sotrage1', 'storage2' et les autres 'logistic_regression' et 'random_forest'
    

    


    


###################### LOGISTIC REGRESSION #######################
    def Logistic_Regression(self, x,y):
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
        d = {'recall': metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]),"confusion": cm,"precision": metrics.precision_score(y_test, y_pred), "neg_precision":cm[1,1]/cm.sum(axis=1)[1], "roc": metrics.roc_auc_score(y_test,y_pred),"class_mod":lr,"method name": 'logistic regression'}
        return d, lr 
####################END OF THE LOGISTIC REGRESSION #############################



##########RANDOM FOREST PROGRAM #######################
######RANDOM FOREST###############
    def random_forest(self, x,y):
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
             "roc": metrics.roc_auc_score(test_labels, rf_predictions),"method name": 'random forest'} #ajouter le modele 
        return d2, model
    
    
    def method_comparison(self, d1,d2):
        rf=0 #compter le nombre de metrics ou la method random forest correspond mieux
        lgr=0 #compter le nombre de metrics ou la regression lineaire correspond mieux 
        for k in d1.keys()-{'class_mod'}-{'confusion'}-{'method name'}:
            if d1[k]>d2[k]:
                if d1['method name']=='linear regression':
                    lgr+=1
                if d1['method name']=='random forest':
                    rf+=1
            else:
                if d2['method name']=='linear regression':
                    lgr+=1
                if d2['method name']=='random forest':
                    rf+=1
        if lgr>rf:
            method='logistic regression'
        else:
            method='random forest'
        return method 
###############END OF THE RANDOM FOREST PROGRAM ###################################



    def main (self):
        storage_data = self.Storages
        c1=0 ###compteur pour la methode logistic regression
        c2=0 ##compteur pour la methode Random forest
        for k, v in storage_data.items():
            dataFrame = storage_data [k]
            
            
            
        # dfply.join.inner_join(dataFrame, price_data)
        
            #deuxieme partie a partir d'ici
            feature_cols = ['NW_Lagged', 'FSW1', 'FSW2']
            x = np.array(dataFrame[feature_cols]) # Features
            y = np.array(dataFrame['NW_b']) # Target variable
            self.model1[k], logistic_regression_model =self.Logistic_Regression(x,y)
            self.model2[k], random_forest_model =self.random_forest(x,y)
            self.coefficients [k] = {'logistic_regression':logistic_regression_model, 'random_forest':random_forest_model }
            c1=0 ###compteur pour la methode logistic regression
            c2=0 ##compteur pour la methode Random forest
            if self.method_comparison(self.model1[k],self.model2[k])=='logistic regression':
                c1+=1
            else:
                c2+=1
            #comparaison des modeles
        if c1>c2:
            print ("The logistic regression is better")
        if c2>c1:
            print ("The random forest is better")
    
        
        
class Regression ():
    def __init__ (self):
        self.Storages = storage_data
        self.dict_regression = dict ()
        self.coefficients = dict ()
    
    def main (self):
        storage_data = self.Storages
        for k, v in storage_data.items():
            dataFrame = storage_data [k]
            dataFrame = dataFrame[dataFrame.NW_b != 0]
            feature_cols = ['NW_Lagged', 'FSW1', 'FSW2']
            global X,y
            X = dataFrame[feature_cols] # Features
            X.dropna()
            y = dataFrame['NW'] # Target variable
        
           
            
            
            #plt.figure(figsize=(15,10))
            #plt.tight_layout()
            #seabornInstance.distplot(dataset['quality'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #probleme: si on met 0.2 au lieu de 0.3, dans le y_train, il n'y a que 1 classe; fit () refuse de marcher
            
            regressor = LinearRegression()  
            l_reg = regressor.fit(X_train, y_train)
            
            #coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
            coeff_df = regressor.coef_
            
            y_pred = regressor.predict(X_test)
            
            df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            df1 = df.head(25)
            
            df1.plot(kind='bar',figsize=(10,8))
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            plt.show()
            
            #np.mean ()
            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))   
            averageValueConsumption = np.mean (y_test)
            maxValueConsumption = np.max (y_test)
            minValueConsumption = np.min (y_test)
            ANRMSE = RMSE/averageValueConsumption
            NRMSE = RMSE/(maxValueConsumption - minValueConsumption)
            r2 = metrics.r2_score(y_test, y_pred)
            corr = scipy.stats.pearsonr(y_test, y_pred)[0]
            d_regression = {'r2': r2, 'rmse': RMSE, 'nrmse': NRMSE, 'anrmse': ANRMSE, 'corr': corr, 'l_reg':l_reg }
            self.dict_regression[k] = d_regression
            self.coefficients[k] = regressor
        
        
        
        
        
X, y = 0,0

def main ():
    seuil = 45
    storage_data = initialisation_data ()
    classification = Classification ()
    regression = Regression ()
    classification.main ()
    regression.main()
    best_model = 'random_forest' #a determiner 
    
    
    
    Supply = pd.DataFrame (data = storage_data['SF - UGS Bierwang']['gasDayStartedOn'])
    Supply['Supply'] = 0
    ###here the foreCasting:
    for k, v in storage_data.items():
            dataFrame = storage_data [k]
            dataFrame = dataFrame[dataFrame.NW_b != 0]
            feature_cols = ['NW_Lagged', 'FSW1', 'FSW2']
            X = dataFrame[feature_cols] # Features
            X.dropna()
    
            
            regressor = classification.coefficients[k][best_model]
            
            #coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
            
            y_pred = regressor.predict(X)
            
            
            
            linear_regressor = regression.coefficients[k]
            NW = linear_regressor.predict (X)
            NW= np.maximum (NW, np.zeros(NW.size))
            dataFrame['NW'] = NW
            f = pd.merge(Supply,dataFrame, on='gasDayStartedOn', how= 'left')
            f.fillna(0, inplace = True)
            f['Supply'] = f['Supply'] + f['NW']
            keys =['gasDayStartedOn', 'Supply']
            Supply = f [keys]
 
    

if __name__ == '__main__':
    main ()