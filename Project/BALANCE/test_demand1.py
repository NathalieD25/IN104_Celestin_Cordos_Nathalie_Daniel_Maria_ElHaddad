import demand
import numpy as np
import pandas as pd
import unittest
import pytest
T=[400,534,600]


class Testconsumption(unittest.TestCase):
    def test_initial_consumption(self):
        obj_1 =demand.consumption(500, -25, 2, 100)
        self.assertEqual(obj_1.a,500)
        self.assertEqual(obj_1.b,-25)
        self.assertEqual(obj_1.c,2)
        self.assertEqual(obj_1.d,100)
        obj_2 =demand.consumption(501.08, -27.4, 3.7, 105.9)
        self.assertEqual(obj_2.a,501.08)
        self.assertEqual(obj_2.b,-27.4)
        self.assertEqual(obj_2.c,3.7)
        self.assertEqual(obj_2.d,105.9)

    def test_no_value(self):
        with pytest.raises(Exception) as e_info:
            obj = MyClass()

    def test_get_consumption (self) :
        A=demand.consumption(56,46,35,24)
        cons1=A.get_consumption([39,543,456])
        self.assertEqual(cons1[0],24)

if __name__=='__main__' :
   unittest.main()
