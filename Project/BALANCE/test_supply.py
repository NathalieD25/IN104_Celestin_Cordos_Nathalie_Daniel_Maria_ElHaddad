import supply
import unittest
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression




class test_Classification (unittest.TestCase) :

    classification = supply.Classification ()
    Stockages = classification.Storages
    keys = list (Stockages.keys ())
    stockage = Stockages[keys[0]]
    feature_cols = ['NW_Lagged', 'FSW1', 'FSW2']
    x = stockage[feature_cols] # Features
    y = np.zeros (x.shape[0])
    for i in range (int ( x.shape[0]/2)):
        y [i] = 1
    
    def test_Supplies (self):
        storage_data = self.Stockages
        self.assertIsInstance (storage_data, dict) #teste si c'est de la bonne forme
        for k,v in storage_data.items():
            self.assertIn ('NW', v.columns)
            self.assertIn ('NW_Lagged', v.columns)
            self.assertIn ('FSW1', v.columns)
            self.assertIn ('FSW2', v.columns)
            self.assertIn ('NW_b', v.columns)
    
    def test_random_forest (self):
        d, lr = self.classification.random_forest (self.x,self.y)
        self.assertIsInstance (d, dict)
        
#    def test_Logistic_Regression (self):
#        d, lr = self.classification.Logistic_Regression (self.x,self.y)
#        self.assertIsInstance (d, dict)
        
    def test_compare_model (self):
        model1 = self.classification.model1 
        model2 = self.classification.model2 
        d1 = {'recall': 0.7956989247311828, 'neg_recall': 0.6491228070175439, 'confusion': np.array([[272,  40],
                    [ 19,  74]], dtype=np.int64), 'precision': 0.6491228070175439, 'neg_precision': 0.7956989247311828, 'roc': 0.8337468982630273, 'class_mod': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False), 'method name': 'logistic regression'}
        d2 = {'recall': 0.6875, 'neg_recall': 0.7051282051282052, 'confusion': np.array([[221,  23],
                [ 25,  55]], dtype=np.int64), 'precision': 0.7051282051282052, 'neg_precision': 0.6875, 'roc': 0.7966188524590164, 'method name': 'random forest'}
        method = self.classification.method_comparison(d1, d2)
        self.assertEqual(method, 'random_forest')
        for k,v in model1.items():
            self.assertIn (k,model2)
            d1 = v
            d2 = model2[k]
            method = self.classification.method_comparison (d1, d2)
            self.assertIn (method,['random_forest', 'logistic_regression'] )
    

class test_Regression ():
    regression = supply.Regression ()
    
    def test_model(self):
        model = self.regression.dict_regression
        for k,v in model.items():
            self.assertIn ('r2',v)
            self.assertIn ('rmse',v)
            self.assertIn ('nrmse',v) 
            self.assertIn ('anrmse',v) 
            self.assertIn ('corr',v)
            self.assertIn ('l_reg',v) 
            





if __name__ == '__main__':
    unittest.main()
      
