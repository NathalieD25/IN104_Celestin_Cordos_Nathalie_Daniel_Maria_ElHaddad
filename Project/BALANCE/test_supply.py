import Consommation
import unittest

class Test(unittest.TestCase) :

   def Test_Logistic_Regression(self) :
      A= Consommation.Classification()
      self.assertEqual(type(Consommation.Logistic_Regression(A,2,3)[0]), dict)
      

   def test_Random_forest (self) :
      A= Consommation.Regression()
      self.assertEqual(type(A.random_forest(A,x)[0]),dict)
      
   
   def test_method_comparison (self) :
      A= Consommation.Classification()
      self.assertEqual(type(methode_comparison(d1,d2)), str)

#   def test_regression (self) :
#      A= Consommation.Regression()
#      self.assertEqual(type(y),dataFrame)


if __name__ == '__main__':
    unittest.main()
      
