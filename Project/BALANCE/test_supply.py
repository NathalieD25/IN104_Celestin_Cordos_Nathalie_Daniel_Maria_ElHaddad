import supply
import unittest
import pandas
import numpy as np

class test_Classification (unittest.TestCase) :

    regression = supply.Classification ()
    items = regression.Storages.items()
    stockage = items [0][1]
    feature_cols = ['NW_Lagged', 'FSW1', 'FSW2']
    x = stockage[feature_cols] # Features
    y = np.zeros (feature_cols.shape[0])
    
    def test_Supplies (self):
        storage_data = self.stockage
        self.assertIsInstance (storage_data, dict) #teste si c'est de la bonne forme
        for k,v in storage_data.items():
            self.assertIn ('NW', v.columns)
            self.assertIn ('NW_Lagged', v.columns)
            self.assertIn ('FSW1', v.columns)
            self.assertIn ('FSW2', v.columns)
            self.assertIn ('NW_b', v.columns)
    
    def test_random_forest (self):
        d, lr = self.regression.random_forest (self.x,self.y)
        self.assertIsInstance (d, dict)
        
    def test_Logistic_Regression (self):
        d, lr = self.regression.Logistic_Regression (self.x,self.y)
        self.assertIsInstance (d, dict)
        
#    def test_compare_model ():

        
        
    







if __name__ == '__main__':
    unittest.main()
      
