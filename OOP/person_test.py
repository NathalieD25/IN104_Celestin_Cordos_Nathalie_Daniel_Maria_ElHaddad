import person
import unittest

class testPerson(unittest.TestCase):               
    cities =   ( 'New York', 'Massachussets', 'Iowa', 'California')           

    def test_move_out(self):           
        for city in self.cities:
            personne = person.Person ('Name', 'Last Name','Knowhere' )
            personne.move_out (city)
            self.assertEqual(personne.state , city)       

class Worker(unittest.TestCase):               
    salaries = ((0,5000, 5000), (500,1000, 1500), (1500,-1700, 0))           

    def test_salary_rise(self):           
        for current, rise, new  in self.salaries:
            worker = person.Worker ('Name', 'Last Name','Knowhere', 'Inventer', current)
            worker.salary_rise (rise)
            self.assertEqual(worker.salary, new)       
            
            
class Student(unittest.TestCase):               
    universities = ('M.I.T.', 'Berkeley', 'Harvard')   
    fields = ('Electronics', 'Computer Engineering', 'Aerospace')        

    def test_change_university(self):           
        for university  in self.universities:
            student = person.Student ("Brandon","Flowers","California","Stanford","Civil Engineering",90)
            student.change_university (university)
            self.assertEqual(student.university, university)     
    def test_change_major (self):
        for field  in self.fields:
            student = person.Student ("Brandon","Flowers","California","Stanford","Civil Engineering",90)
            student.change_major (field)
            self.assertEqual(student.major, field)  
        

if __name__ == '__main__':
    unittest.main()