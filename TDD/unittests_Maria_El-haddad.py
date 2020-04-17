import unittest
from TD2 import Car, OffRoader, Roadster

class Test(unittest.TestCase) :

   def test_turnOff(self) : 
      Car1=Car('Infiniti','QX','30' )
      Car2=Car(' Mercedes-Benz','E-class', '200')
      Car1.turnOff()
      Car2.turnOff()
      self.assertEqual(Car1.status, 0 )
      self.assertEqual(Car2.status, 0 )
      
   def test_roofOn(self) :
      Ferrari= Roadster(' Ferrari' ,' 488', 488)
      Maserati= Roadster(' Maserati', ' Mistral' , 3500)
      Ferrari.roofOn()
      Maserati.roofOn()
      self.assertEqual(Ferrari.roof,1)
      self.assertEqual(Maserati.roof,1)

   def test_lowerVehicule(self) :
      OffRoader1 = OffRoader (' Range Rover' , ' Defender' , ' def-1' , 0.2921, 0.2159)
      OffRoader2 = OffRoader(' Jeep' , ' Wrangler' , 'W-1' , 0.274, 0.246)
      OffRoader1.lowerVehicle(0.2)
      OffRoader2.lowerVehicle(0.2)
      self.assertEqual(OffRoader1.groundClearance,0.2159)
      self.assertEqual(OffRoader2.groundClearance,0.246)

   def test_sportModeoff(self) :
      Ferrari= Roadster(' Ferrari' ,' 488', 488)
      Maserati= Roadster(' Maserati', ' Mistral' , 3500)
      Ferrari.sportModeOff()
      Maserati.sportModeOff()
      self.assertEqual( Ferrari.sportMode, 0)
      self.assertEqual( Maserati.sportMode, 0)

if __name__=='__main__':
      unittest.main()
      
   
