import numpy as np
import pandas as pd
import math
from collections import deque

#Task#1
#Calculate sum by a function of n and d
def calculateSum(n, d):

	result = 0
	for i in range(0, 1000):
		d = pow(2, i)
		if int(n/d) != 0:
			result = result + int(n/d)
	return result
result = calculateSum(25, 1)
print(result)
print()


# A utility function to print an
# array p[] of size 'n'
def generateSumUtility(p, n):
	for i in range(0, n):
		print(p[i], end=" ")
	print()

#Task 2: Generate all summatations for a particular number n, using conquer and divide method
def generateSum(n):
	p = [0] * n
	k = 0
	p[k] = n
	while True:
		generateSumUtility(p, k + 1)
		rem = 0
		while k >= 0 and p[k] == 1:
			rem += p[k]
			k -= 1

		if k < 0:
			print()
			return

		p[k] -= 1
		rem += 1
		while rem > p[k]:
			p[k + 1] = p[k]
			rem = rem - p[k]
			k += 1

		p[k + 1] = rem
		k += 1
		
generateSum(5)

#Task 3: Josephus circle using dynammic programming method
def Josephus(n, k):

	ls = range(1, n+1)
	d = deque(ls)
	while len(d) > 1:
		d.rotate(-k)
		print(d.pop())
	print('survivor:', d[0])

Josephus(7, 3)
