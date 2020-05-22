# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:56:30 2020

@author: Celestin Cordos
"""

import balance
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class test_balance (unittest.TestCase) :
    
    def test_market_decision (self):
        n = 201
        Supply = [i   for i in range (n)]
        Demand = [n-1-i for i in range (n)]
        d = {'Supply': Supply, 'Demand': Demand, 'Supply_real': Supply, 'Demand_real':Demand}
        DF = pd.DataFrame(data = d)
        balance.market_decision (DF)
        decision = DF['Decision']
        real_decision = DF['Decision_real']
        solution = ['BUY' for i in range (int((n-1)/2))]
        solution.append ('FLAT')
        solution = solution + ['SELL' for i in range( int(((n-1)/2)))]
        self.assertEqual (decision, solution)
        self.assertEqual (real_decision, solution)




if __name__ == '__main__':
    unittest.main()