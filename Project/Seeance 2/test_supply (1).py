import Consommation.modified.py

class Test(unittest.TestCase) :

   def Test_Logistic_Regression(self) :
      A= Classification(storage_data,{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} },{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} })
      self.assertEqual(type(Logistic_Regression(A,2,3)[0]), dict)
      

   def test_Random_forest (self) :
      A= Classification(storage_data,{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} },{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} })
      self.assertEqual(type(random_forest(A,x)[0]),dict)
      
   
   def test_method_comparison (self) :
      A= Classification(storage_data,{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} },{"SF - UGS Rehden": {} ,"SF - UGS Kraak": {},"SF - UGS Stassfut" :{},"SF - UGS Harsefeld" :{},"SF - UGS Breitburnn" : {}, "SF - UGS Epe Uniper H-Gas" : {}, "SF - UGS Eschenfelden" : {}, "SF - UGS Inzeham-West ": {}, "SF - UGS Bierwang" : {}, "SF - UGS Jemgum H (EWE)" : {}, "SF - UGS Peckensen" : {}, " SF - UGS Peckensen " : {}, " SF  -UGS Etzel ESE (Uniper Ener) " : {} })
      self.assertEqual(type(methode_comparison(A,d1,d2)), str)

   def test_regression (self) :
      A= Regression(storage_data,dict(),dict())
      self.assertEqual(type(y),dataFrame)


if __name__ == '__main__':
    unittest.main()
      
