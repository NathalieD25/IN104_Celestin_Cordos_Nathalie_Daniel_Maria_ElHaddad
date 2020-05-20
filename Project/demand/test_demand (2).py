import consumption_script_to_fill_modified.py
import numpy as np 
import pandas as pd 

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
      ga, gb, gc, gd = 900,  -35,  6, 300
      f=pd.DataFrame()
      f['Date'] = [20/12/2015,17/7/2016]
      f[' LDZ'] = [655,666]
      f['Actual'] = [535,653]
      f['Normal'] = [654,534]
      A= optimize_sigmoid(f)
      coef,cov = curve_fit(h,list (A.f['Actual']),list(A.f['LDZ']),[A.ga,A.gb, A.gc, A.gd])
      self.assertEqual(coef,A.__coef)
      self.assertEqual(cov, A.__cov) 
      s= consumption_sigmoid( list(A.f['Actual' ]), list(A.f['LDZ']),coef[0],coef[1],coef[2],coef[3], plot=True)
      (va,vb,vc,vd)= get_fit_metrics(s, A.f['LDZ'])
      self.assertEqual(hasattr(va,'A.__corr'),True)
      self.assertEqual(hasattr(vb,'A.__rmse'),True)
      self.assertEqual(hasattr(vc,'A.__nrmse'),True)
      self.assertEqual(hasattr(vd,'A.__anrmse'),True)
      
   def test__fit_metrics(self) :
      f=pd.DataFrame()
      f['Date'] = [20/12/2015,17/7/2016]
      f[' LDZ'] = [655,666]
      f['Actual'] = [535,653]
      f['Normal'] : [654,534]
      A= optimize_sigmoid(f)
      if A.__corr is not None :
         self.assertEqual(A.__corr, A.fit_metrics[0])
         self.assertEqual(A.__rmse, A.fit_metrics[1])
         self.assertEqual(A.__nrmse, A.fit_metrics[2])
         self.assertEqual(A.__anrmse, A.fit_metrics[3])

   def test_creat_consumption(self) :
      f=pd.DataFrame()
      f['Date'] = [20/12/2015,17/7/2016]
      f[' LDZ'] = [655,666]
      f['Actual'] = [535,653]
      f['Normal'] : [654,534]
      A= optimize_sigmoid(f)
      if A.__corr is not None : 
         con=consumption(A.__coef[0],A.__coef[1],A.__coef[2],A.__coef[3])
         self.assertEqual(creat_consumtion(A),con)

   
   
      


 
      
