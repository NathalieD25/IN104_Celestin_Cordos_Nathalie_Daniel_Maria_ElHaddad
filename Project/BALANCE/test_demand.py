import DEMAND
import numpy as np 
import pandas as pd 
import unittest
import pytest
T=[400,534,600]


class Testconsumption(unittest.TestCase):
    def test_initial_consumption(self):
        obj_1 =DEMAND.consumption(500, -25, 2, 100)    
        self.assertEqual(obj_1.a,500)
        self.assertEqual(obj_1.b,-25)
        self.assertEqual(obj_1.c,2)
        self.assertEqual(obj_1.d,100)
        obj_2 =DEMAND.consumption(501.08, -27.4, 3.7, 105.9)
        self.assertEqual(obj_2.a,501.08)
        self.assertEqual(obj_2.b,-27.4)
        self.assertEqual(obj_2.c,3.7)
        self.assertEqual(obj_2.d,105.9)
        
    def test_no_value(self):
        with pytest.raises(Exception) as e_info:
            obj = MyClass()
# 
###Not working
# class Testoptimize_sigmoid(unittest.TestCase):
#     
#     def test_initial_optimize_sigmoid(self):
#         f=pd.DataFrame()
#         f['Date'] = [20/12/2015,17/7/2016]
#         f[' LDZ'] = [655,666]
#         f['Actual'] = [535,653]
#         f['Normal'] : [654,534]
#         obj_3 = demand1.optimize_sigmoid(f)
#         self.assertEqual(obj_3.f,f)





class Test(unittest.TestCase):
    
    def test_afunction_throws_exception(self):
        test_df= pd.DataFrame(columns=['Date', 'LDZ', 'Actual','Normal'])
        self.assertRaises(TypeError,lambda:DEMAND.scatter_plot(test_df,x= "Actual",y="LDZ",col=1))
        self.assertRaises(TypeError,lambda:DEMAND.scatter_plot(0,x= "Actual",y="LDZ",col='red'))



# class Test(unittest.TestCase) :
#   
#    def test_get_consumption (self) :
#       A=consumption('1','2','3','4')
#       cons=[]
#       for i in T :
#          cons.append(A.d+A.a/(1+(A.b/(T[i]-40))**A.c))
#       pam = get_consumption(A,T)
#       self.assertEqual(pam,cons)
   
   # def test_sigmoide (self) :
   #    A=consumption('1','2','3','4')
   #    temp=[i for i in range (-40,40,1)]
   #    sig=consumption_sigmoid(temp,None, A.a, A.b, A.c, A.d,1)
   #    self.assertEqual(sig,sigmoid(A,1))
   # 
   # def test_optimize ( self) :
   #    ga, gb, gc, gd = 900,  -35,  6, 300
   #    f=pd.DataFrame()
   #    f['Date'] = [20/12/2015,17/7/2016]
   #    f[' LDZ'] = [655,666]
   #    f['Actual'] = [535,653]
   #    f['Normal'] = [654,534]
   #    A= optimize_sigmoid(f)
   #    coef,cov = curve_fit(h,list (A.f['Actual']),list(A.f['LDZ']),[A.ga,A.gb, A.gc, A.gd])
   #    self.assertEqual(coef,A.__coef)
   #    self.assertEqual(cov, A.__cov) 
   #    s= consumption_sigmoid( list(A.f['Actual' ]), list(A.f['LDZ']),coef[0],coef[1],coef[2],coef[3], plot=True)
   #    (va,vb,vc,vd)= get_fit_metrics(s, A.f['LDZ'])
   #    self.assertEqual(hasattr(va,'A.__corr'),True)
   #    self.assertEqual(hasattr(vb,'A.__rmse'),True)
   #    self.assertEqual(hasattr(vc,'A.__nrmse'),True)
   #    self.assertEqual(hasattr(vd,'A.__anrmse'),True)
   #    
   # def test__fit_metrics(self) :
   #    f=pd.DataFrame()
   #    f['Date'] = [20/12/2015,17/7/2016]
   #    f[' LDZ'] = [655,666]
   #    f['Actual'] = [535,653]
   #    f['Normal'] : [654,534]
   #    A= optimize_sigmoid(f)
   #    if A.__corr is not None :
   #       self.assertEqual(A.__corr, A.fit_metrics[0])
   #       self.assertEqual(A.__rmse, A.fit_metrics[1])
   #       self.assertEqual(A.__nrmse, A.fit_metrics[2])
   #       self.assertEqual(A.__anrmse, A.fit_metrics[3])
   # 
   # def test_creat_consumption(self) :
   #    f=pd.DataFrame()
   #    f['Date'] = [20/12/2015,17/7/2016]
   #    f[' LDZ'] = [655,666]
   #    f['Actual'] = [535,653]
   #    f['Normal'] : [654,534]
   #    A= optimize_sigmoid(f)
   #    if A.__corr is not None : 
   #       con=consumption(A.__coef[0],A.__coef[1],A.__coef[2],A.__coef[3])
   #       self.assertEqual(creat_consumtion(A),con)

if __name__=='__main__' :
    unittest.main()

   
