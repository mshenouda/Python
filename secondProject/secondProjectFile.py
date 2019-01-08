import sys
import io
import string
import os

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

loopFunc()
classFunc()