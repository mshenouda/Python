import numpy
import pandas

# Definition for singly-linked list.
class ListNode:
	def __init__(self, val=0, next=None):
         self.val = val
         self.next = next


def findNumber(arr, k):
	s = set(arr)
	if s.__contains__(k):
		return 'YES'
	return 'NO'

#l1 = [2, 3, 5]
#result = findNumber(l1, 1)
#print(result)

def oddNumbers(l, r):
	# Write your code here

	s = set()
	if l % 2 == 0:
		l = l + 1

	if r % 2 == 0:
		r = r - 1

	while l <= r:
		s.add(l)
		l = l + 2

	result = list(s)
	result.sort()
	return result

result = oddNumbers(3, 9)
print(result)