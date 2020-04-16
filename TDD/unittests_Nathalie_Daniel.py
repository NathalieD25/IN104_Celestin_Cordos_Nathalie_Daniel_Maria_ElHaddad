import unittest

from TD2 import Car,OffRoader,Roadster

class TestTD2(unittest.TestCase):
	
	def test_turnOn(self):
		Car1=Car('Mercedes-Benz','S-Class','S-1')
		Car2=Car('Audi','A6','A6-1')
		Car1.turnOn()
		Car2.turnOn()
		self.assertEqual(Car1.status,1)
		self.assertEqual(Car2.status,1)


	def test_roofOff(self):
		Roadster1 = Roadster ('Ferrari', '488', '488-1' )
		Roadster2 = Roadster ('Lammborghini', 'Aventador', 'A-1')
		Roadster1.roofOff ()
		Roadster2.roofOff()
		self.assertEqual(Roadster1.roof,0)
		self.assertEqual(Roadster2.roof,0)

	def test_raiseVehicule(self):
		OffRoader1 = OffRoader ('Range Rover', 'Defender', 'def-1', 0.2921, 0.2159)
		OffRoader2 = OffRoader ('Jeep', 'Wrangler','W-1', 0.274, 0.246 )
		OffRoader.raiseVehicle(OffRoader1,0.20)
		OffRoader.raiseVehicle(OffRoader2,0.015)
		self.assertEqual(OffRoader1.groundClearance,OffRoader1.groundClearance)
		self.assertEqual(OffRoader2.groundClearance,0.261)
	def test_sportModeOn(self):
		Roadster1 = Roadster ('Ferrari', '488', '488-1' )
		Roadster2 = Roadster ('Lammborghini', 'Aventador', 'A-1')
		Roadster1.sportModeOn ()
		Roadster2.sportModeOn()
		self.assertEqual(Roadster1.sportMode,1)
		self.assertEqual(Roadster2.sportMode,1)



if __name__=='__main__':
    unittest.main()

