import consumption_script_to_fill_modified.py
import numpy as np 

T=[400,534,600]

class Test(unittest.TestCase) :
  
   def test_get_consumption (self) :
      A=consumption('1','2','3','4')
      cons=[]
      for i in T :
          cons.append(A.d+A.a/(1+(A.b/(T[i]-40))**A.c))
      pam = get_consumption(A,T)
      self.assertEqual(pam,cons)
   
   def test_sigmoide (self) :
      A=consumption('1','2','3','4')
      temp=[i for i in range (-40,40,1)]
      sig=consumption_sigmoid(temp,None, A.a, A.b, A.c, A.d,1)
      self.assertEqual(sig,sigmoid(A,1))
   
   def test_optimize ( self) :
      __guess_a, __guess_b, __guess_c, __guess_d = 900,  -35,  6, 300
      A= optimize_sigmoid( pd.DataFrame ( {'Date' : [20/12/2015,17/07/2016], ' LDZ' = [655,666], 'Actual' : [535,653], 'Normal' : [654,534]})
      coef , cov = curve_fit(h,list (A.f['Actual']), list(A.f['LDZ']),[__guess_a,__guess_b, __guess_c, __guess_d])
      self.assertEqual(coef,A.__coef)
      self.assertEqual(cov, A.__cov) 
      s= consumption_sigmoid( list(A.f['Actual' ], list(A.f['LDZ']),coef[0],coef[1],coef[3], plot=True)
      a,b,c,d= get_fit_metrics(s,A.f['LDZ'])
      hasattr(a,A.__corr)
      hasattr(b,A.__rmse)
      hasattr(c,A.__nrmse)
      hasattr(d,A.__anrmse)
      
   def test__fit_metrics(self) :
      A= optimize_sigmoid( pd.DataFrame ( {'Date' : [20/12/2015,17/07/2016], ' LDZ' = [655,666], 'Actual' : [535,653], 'Normal' : [654,534]})
      if A.__corr is not None :
         self.assertEqual(A.__corr, A.fit_metrics[0])
         self.assertEqual(A.__rmse, A.fit_metrics[1])
         self.assertEqual(A.__nrmse, A.fit_metrics[2])
         self.assertEqual(A.__anrmse, A.fit_metrics[3])

   def test_creat_consumption(self) :
      A= optimize_sigmoid( pd.DataFrame ( {'Date' : [20/12/2015,17/07/2016], ' LDZ' = [655,666], 'Actual' : [53
      if A.__corr is not None : 
         con=consumption(A.__coef[0],A.__coef[1],A.__coef[2],A.__coef[3])
         self.assertEqual(creat_consumtion(A),con)

   
   
      


 
      
