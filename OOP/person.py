#Our superclass will be Person which contains 3 attributes (firstname,lastname and state (which state is the person living in)) that are common to our 2 subclasses: Student and Worker
class Person:
    def __init__(self,firstname,lastname,state):
        self.firstname=firstname
        self.lastname=lastname
        self.state=state
    def move_out(self,new_state):
        self.state=new_state
        print('%s is now living in %s' % (self.firstname,self.state))

class Worker(Person): #Defining our first subclass with it functions
     def __init__(self,firstname,lastname,state,field,salary):
         super().__init__(firstname,lastname,state)
         self.field=field
         self.salary=salary
     def get_field(self):
         return self.field
     def salary_rise(self,add_to_salary):
         self.salary+=add_to_salary
         print ('Salary rise! %s %s is now earning %d $' %(self.firstname,self.lastname,self.salary))
     def change_career(self,new_field):
         self.field=new_field
         print ('%s %s is now a %s' %(self.firstname,self.lastname,self.field) )

class Student(Person): #Defining our 2nd subclass with it functions
    def __init__(self,firstname,lastname,state,university,major,grade):
         self.major=major
         self.university=university
         self.grade=grade
         super().__init__(firstname,lastname,state)
    def change_university(self,new_university):
         self.university=new_university
         print ('%s %s is now pursuing his/her major at %s' %(self.firstname,self.lastname,self.university))
    def change_major(self,new_major):
        self.major=new_major
        print ('%s %s has now enrolled in a new major: %s' %(self.firstname,self.lastname,self.major))
    def get_grade(self):
        return self.grade
#Our main program in order to test the superclass and subclasses that were defined
def main():
#First test for our 2nd subclass (Student)
    firststudent=Student("Brandon","Flowers","California","Stanford","Civil Engineering",90)
    print (firststudent.firstname)
    Person.move_out(firststudent,"Massachussets")
    Student.change_university(firststudent,"MIT")
    Student.change_major(firststudent,"Mechanical Engineering")
    print ('The student grade is : %d over 100' %(Student.get_grade(firststudent)))
#Second test for our 1st subclass (Worker)
    firstworker=Worker("Ellen","Pompeo","Seattle","Actress",500000)
    print('%s %s is an %s' %(firstworker.firstname,firstworker.lastname,Worker.get_field(firstworker)))
    Person.move_out(firstworker,"New York")
    Worker.change_career(firstworker,"singer")
    Worker.salary_rise(firstworker,200000)