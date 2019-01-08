import sys
import os
import string
import numbers


def stringFunc(firstName,lastName):
	print('My firstName:' + firstName + ' , lastName: '+lastName)
	print('My firstName: {}, my lastName: {}'.format((firstName).lower(),(lastName).capitalize()))
	print('My firstName: {}, my lastName: {}'.format((firstName).isupper(),(lastName).islower()))


stringFunc('MINA','SHENOUDA')



