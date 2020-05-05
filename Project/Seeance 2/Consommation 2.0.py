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
from sklearn.metrics import confusion_matrix
import numpy as np



seuil =45

storage_data = pd.read_excel("storage_data.xlsx", sheet_name=None)
price_data = pd.read_csv('price_data.csv')
price_data.rename(columns={'Date':'gasDayStartedOn '}, inplace=True)

#creating empty dictionnaries
model1=dict() #for the logistic regression 
model2=dict() #for the random forest

lr = LogisticRegression()

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
        X = dataFrame[feature_cols] # Features
        y = dataFrame['Net Withdrawal binary'] # Target variable
        
    
        
        lr.fit(X, y)
        
        print(lr.coef_)
        print(lr.intercept_)
        
        
X_test = 
#y_pred = lr.predict(x_test)
#
#confusion_matrix(y_test, y_pred)
#
#
#df = pd.DataFrame({'x': x_test[:,0], 'y': y_test})
#df = df.sort_values(by='x')
#from scipy.special import expit
#sigmoid_function = expit(df['x'] * lr.coef_[0][0] + lr.intercept_[0]).ravel()
#plt.plot(df['x'], sigmoid_function)
#plt.scatter(df['x'], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')

##Start of the random forest method 
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    RSEED = 50


# 30% examples in test data
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

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
    train_rf_predictions = model.predict(train)
    train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
    rf_predictions = model.predict(test)
    rf_probs = model.predict_proba(test)[:, 1]

    from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

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

        plt.figure(figsize = (8, 6))
        plt.rcParams['font.size'] = 16

    # Plot both curves
        plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
        plt.plot(model_fpr, model_tpr, 'r', label = 'model')
        plt.legend();
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
        plt.show();

        evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
        plt.savefig('roc_auc_curve.png')

    from sklearn.metrics import confusion_matrix
    import itertools

    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    # """
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

    # Plot the confusion matrix
        plt.figure(figsize = (10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size = 24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size = 14)
        plt.yticks(tick_marks, classes, size = 14)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

    # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18)
        plt.xlabel('Predicted label', size = 18)

# Confusion matrix
    cm2= confusion_matrix(test_labels, rf_predictions)
    d2= {"recall": metrics.recall_score(test_labels, rf_predictions), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]), "confusion": cm,"precision": metrics.precision_score(test_labels, rf_predictions), "neg_precision":cm[1,1]/cm.sum(axis=1)[1], "roc": metrics.roc_auc_score(test_labels, rf_predictions)}
