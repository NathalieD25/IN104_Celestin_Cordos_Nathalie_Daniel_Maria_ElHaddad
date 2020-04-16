# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:17:58 2020

@author: Celestin Cordos
"""

#All units S.I.

class Car :
    def __init__ (self, brand, model, number): #status = 1 if the car is turned on, 0 otherwise
        self.brand = brand
        self.model = model
        self.status = 0
        self.serialNumber = number
        
    def turnOn (self):
        self.status = 1
    
    def turnOff (self):
        self.status = 0
        
        
class OffRoader (Car) :
    def __init__ (self, brand, model, number, maxHeight, minHeight): 
        Car.__init__ (self, brand, model, number)
        self.maxHeight = maxHeight
        self.minHeight = minHeight
        self.groundClearance = minHeight
    
    def raiseVehicle (self, height): #if height is 2 mm, the vehicle raises by 2mm...
        self.groundClearance = min (self.maxHeight, self.groundClearance + height)
        print (self.serialNumber, ' : ',"Vehicle raised. New ground clearance:", self.groundClearance)
        
    def lowerVehicle (self, height):
        self.groundClearance = max (self.minHeight, self.groundClearance - height)
        print (self.serialNumber, ' : ',"Vehicle lowered. New ground clearance:", self.groundClearance)
        
class Roadster (Car):
    def __init__ (self, brand, model, number): #sportMode and roof: 1 foor on, 0 for off
        Car.__init__ (self, brand, model, number)
        self.sportMode = 0
        self.roof = 1
        
    def sportModeOn (self):
        self.sportMode = 1
        print (self.serialNumber, ' : ', 'sport mode on')
    
    def sportModeOff (self):
        self.sportMode = 0
        print (self.serialNumber, ' : ','sport mode off')
        
    def roofOn (self):
        self.roof = 1
        print (self.serialNumber, ' : ','roof on')
        
    def roofOff (self):
        self.roof = 0
        print (self.serialNumber, ' : ','roof off')
        
        
defender_1 = OffRoader ('Range Rover', 'Defender', 'def-1', 0.2921, 0.2159)
wrangler_1 = OffRoader ('Jeep', 'Wrangler','W-1', 0.274, 0.246 )

spyder_488_1 = Roadster ('Ferrari', '488', '488-1' )
aventador_1 = Roadster ('Lammborghini', 'Aventador', 'A-1')


defender_1.raiseVehicle (0.30)
wrangler_1.raiseVehicle (0.01)
defender_1.lowerVehicle (0.05)

spyder_488_1.roofOff ()
spyder_488_1.sportModeOn ()

aventador_1.roofOff ()
