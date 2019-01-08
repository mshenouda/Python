import sys
import os
import string
import numbers


def stringFunc(firstName,lastName):
	print('My firstName:' + firstName + ' , lastName: '+lastName)
	print('My firstName: {}, my lastName: {}'.format((firstName).lower(),(lastName).capitalize()))
	print('My firstName: {}, my lastName: {}'.format((firstName).isupper(),(lastName).islower()))


def loopFunc():
		#Typing for loops
		myList=['mina','hanna','morris']

		for i in range(1,5):
			print ('Number = '+str(i))


stringFunc('MINA','SHENOUDA')
loopFunc()



