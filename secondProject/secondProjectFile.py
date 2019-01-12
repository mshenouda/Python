import sys
import io
import string
import os
import datetime
import time
from collections import namedtuple


#starting for loop and while loop functions
def loopFunc():
	#list functions
	mylist=['mina','hanna','morris']
	mystr=''

	for i in range(1,5):
		print('Numbers: '+str(i))

	for i in mylist:
		mystr+='{} '.format(i.upper())
	print('My name is {}'.format(mystr))

	#key value pairs
	mydict = {"name":"MINA","age":36,"marital":"married"}
	keys=""
	values=""

	for key,value in mydict.items():
		keys+=key.capitalize()+" "
		values+=str(value)+" "

	print('Keys of the mydict : '+keys)
	print('Values of the mydict : '+values)

def classFunc():
	class employee:
		def __init__(self,firstName="",lastName="",age=0,marital=""):
			self.firstName=firstName
			self.lastName=lastName
			self.age=age
			self.marital=marital

		def __str__(self):
			return '{} {} of {} is {}'.format(self.firstName.capitalize(),self.lastName.capitalize(),str(self.age),self.marital.upper())

	class manager(employee):
		def __init__(self,firstName="",lastName="",age=0,marital="",position=""):
			employee.__init__(self,firstName,lastName,age,marital)
			self.position=position

		def __str__(self):
			return employee.__str__(self)+' has position of '+self.position.upper()

	emp_Mina = employee('mina', 'shenouda', 35, 'divorced')
	emp_Meret = employee('meret', 'shenouda', 31, 'married')
	emp_Marcos = employee('marcos', 'shenouda', 33, 'married')
	print(emp_Mina)
	print(emp_Marcos)
	print(emp_Meret)

	man_Mina=manager(emp_Mina.firstName,emp_Mina.lastName,emp_Mina.age,emp_Mina.marital,'ceo')
	print(man_Mina)

def listNamedTubles():
	#dictionary
	namedTuble = ['red', 'green', 'blue', 'orange']
	namedTuble.extend(['sorry','my baby'])
	print(namedTuble)
	print(namedTuble.index('orange'))
	namedTuble_str = ', '.join(namedTuble)
	print(namedTuble_str)
	mylist = namedTuble_str.split(', ')
	print(mylist)

	# for num, item in enumerate(namedTuble, start=1):
	# 	print(type(item))
	# 	print(num, item)

	#Tubles are immutable
	myTuble = ('mina','hanna','morris')
	print(myTuble)


def func():
	def printHello(greeting, name='You'):
		return '{} - {}'.format(greeting,name)

	def funcArgsKwargs(*args,**kwargs):
		print(' Args are: ', args)
		print(' Kwargs are: ', kwargs)

	months = [31,28,31,30,31,30,31,31,30,31,30,31]
	def isleap(year):
		return (year % 4 == 0 and year % 100 != 0 or year % 400 == 0)

	def month_in_year(year,month):
		month -= 1
		if (month == 1 and isleap(year)):
			return 29
		else:
			return months[month]

	print(printHello('Hi','Mina').upper())
	funcArgsKwargs(*['Mina', 'Hanna'])
	print(month_in_year(2000,2))

def TryExcept():
	Person = {'name': 'mina', 'age': 35, 'gender': 'male'}
	print(Person)

	try:
		print('Person of {name} and age {age} is {jub}'.format(**Person))
	except KeyError as e:
		print('Error message '+str(e))

	mylist = ['red','green','blue']
	try:
		print('colors of the list are {}'.format(mylist[e]))
	except IndexError as e:
		print('Error message '+str(e))
	except Exception:
		print('Unkown exception')
	finally:
		print('End of the exception code')

def generators(mylist):
	for i in mylist:
		yield(i**2)

# mylist = [ 1,2,3,4,5,6 ]
# print(mylist)
#
# nums = generators(mylist)
# for num in nums:
# 	print(num)


def namedTuples():
	Color = namedtuple('Color',['red','green','blue'])
	color = Color(55,255,26)
	print(color.red)
	print(color.green)

#namedTuples()

def decoratorFunc(original_function):
	def wrapper():
		print("Execute this code before {}".format(original_function.__name__))
		return original_function()
	return wrapper

class decoratorClass(object):

	def __init__(self,original_function):
		self.original_function = original_function

	def __call__(self, *args, **kwargs):
		print("Execute this code before {}".format(self.original_function.__name__))
		return self.original_function(*args, **kwargs)

@decoratorClass
def display():
	print("This is a display function")

#display()

def elseClauseWithLoops():
	mylist = []
	for i in range(1,10,2):
		mylist.append(i)
	else:
		print('Index is incorrect !!')
		print(mylist)



#loopFunc()
#classFunc()
#listNamedTubles()
#func()
#TryExcept()
#generators()
#elseClauseWithLoops()


#display()

# def addition(*args, **kwargs):
# 	return "This is function with {}".format(addition.__name__)
#
#
# def multiply(*args, **kwargs):
# 	return "This is function with {}".format(multiply.__name__)

#displayFunc = decoratorFunc(display)
#displayFunc()



class Employee():
	raiseAmount = 1.04
	profit = 0
	def __init__  (self , firstName , lastName , age):
		self.firstName = firstName
		self.lastName = lastName
		self.age = age

	@property
	def email(self):
		return f'{self.firstName}.{self.lastName}+"@email.com"'

	@property
	def fullname(self):
		return f'{self.firstName}-{self.lastName}'

	@fullname.setter
	def fullname(self,fullname):
		firstName, lastName = fullname.split('-')
		self.firstName = firstName
		self.lastName = lastName


	@fullname.deleter
	def fullname(self):
		print('Deleting fullName !!!')
		self.firstName = None
		self.lastName = None

	def apply_raise(self,raiseAmount):
		self.raiseAmount = raiseAmount
		return self.raiseAmount

	@classmethod
	def fromString(cls,fullStr):
		firstName,lastName,age = fullStr.split("-")
		return cls(firstName,lastName,age)

	@staticmethod
	def isWorkingDay(day):
		if (day.weekday() == 5 or day.weekday() == 6):
			return False
		return True

class Developer(Employee):
	def __init__(self, firstName, lastName, age, title):
		Employee.__init__(self, firstName, lastName, age)
		self.title = title

	@classmethod
	def fromString(cls, fullStr):
		firstName, lastName, age, title = fullStr.split("-")
		return cls(firstName, lastName, age, title)

	def printInfo(self):
		print('{}-{} of age {} with email {} has title {}'.format(self.firstName,self.lastName,self.age,self.email,self.title))


array1 = [1,3,5,7]
array2 = [2,4,6,8]

def addition(x, y):
	return x+y

def multiply(x, y):
	return x*y

def subtract(x, y):
	return x-y

def functionDecorator (original_function, x, y):

	def wrapper(*args, **kwargs):
		args = (x, y)
		return original_function(*args, **kwargs)
	return wrapper

def callDecorator(function,array1,array2):
	for i in range(0,len(array1)):
		result = functionDecorator(function, array1[i], array2[i])
		print(result())


callDecorator(addition, array1, array2)

# mina_emp = Employee("Mina","Shenouda",36)
# meret_emp = Employee("Meret","Hanna",30)
#
# mina_emp.firstName = 'Samir'
# print(mina_emp.email)
# print(mina_emp.fullname)
#
# mina_emp.fullname = 'Bahget-Samir'
# print(mina_emp.email)
# del mina_emp.fullname
# print(mina_emp.fullname)

# print(mina_emp.email)
# print(Employee.raiseAmount)
# print(mina_emp.apply_raise(1.03))
# print(meret_emp.raiseAmount)
#
# myname = "sheno-ramy-37"
# newEmp = Employee.fromString(myname)
# print(newEmp.email)
#
# myDate = datetime.date(2018,10,22)
# print(Employee.isWorkingDay(myDate))

# firstDevStr = "Trevor-Serag-35"
# firstDevTitle ="TeadLead"
# firstDevStr = firstDevStr + "-"+firstDevTitle
# firstDev = Developer.fromString(firstDevStr)
# firstDev.printInfo()

