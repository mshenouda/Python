from collections import OrderedDict
import math
from itertools import permutations
import re
from collections import defaultdict
import string
from collections import Counter
import random
import queue
from functools import reduce

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def print2D (_2dList):
	for elem in _2dList:
		print(elem)

def printDict(myDict):
	for key, value in myDict.items():
		print(key, value)

def minimumAbsDifference(arr):

	arr.sort()
	diff = 1000
	result = list(list())

	for i in range(0, len(arr)-1):

		tmp = arr[i+1] - arr[i]
		if (tmp > diff):
			continue

		if (tmp < diff):
			if (len(result) > 0):
				result.clear()
		diff = min(tmp, diff)
		tmpList = list()
		tmpList.append(arr[i])
		tmpList.append(arr[i+1])
		result.append(tmpList)

	return result

# #arr = [4, 2, 1, 3]
# #arr = [1,3,6,10,15]
# arr = [3,8,-10,23,19,-4,-14,27]
# result = minimumAbsDifference(arr)
# print2D(result)


def numberOfBoomerangs(points):

	mySet = set()
	for point in points:
		mySet.add(point[0])
	myList = list(mySet)
	counter = 0
	for i in range(1, len(myList)-1):
		if ( abs(myList[ i + 1] - myList[i]) == abs(myList[ i ] - myList[i - 1]) ):
			counter += 2

	return counter

# points = [[0, 0], [1, 0], [2, 0]]
# val = numberOfBoomerangs(points)
# print(val)

def sortString(s):

	d = defaultdict(int)
	for k in s:
		d[k] += 1

	od = OrderedDict(sorted(d.items()))
	result = str()
	stack = []
	flag = False
	print(od)

	counter = 0
	while (counter < len(s)) :

		for k, v in od.items():
			if ( v > 0 ):
				counter = counter + 1
				od[k] = od[k] - 1
				if (flag == False):
					result = result + k
				else:
					stack.append(k)

		if (flag == True):
			while stack.__len__() > 0:
				result = result + stack.pop()
		flag = not flag
	return result

# s = "aaaabbbbcccc"
# #s = "rat"
# #s = "leetcode"
# #s = "spo"
# result = sortString(s)
# print(result)

def generateTheString(n):

	result = str()
	if (n % 2 == 1):
		for i in range(0, n):
			result = result + 'a'
	else:
		for i in range(0, n-1):
			result = result + 'a'
		result = result + 'b'
	return result

# n = 8
# result = generateTheString(n)
# print(result)

def hammingDistance(x, y):

	result = bin(x.__xor__(y))
	counter = result.count('1')
	return counter

# x = 2
# y = 4
# n = hammingDistance(x, y)
# print(n)

def getDigits(num):

	myList = list()
	s = num.__str__()
	for a in s:
		myList.append(int(a))
	return myList

def checkDivide(num, myList):

	for i in myList:
		if (i == 0):
			return False

		if (num % i != 0):
			return False
	return True

def selfDividingNumbers(left, right):

	selfNums = list()

	for i in range(left, right+1):
		tmpList = list()
		if (i != 0 & i <= 9):
			tmpList = getDigits(i)
			if (checkDivide(i, tmpList)):
				selfNums.append(i)

	return selfNums

# left = 1
# right = 22
# myList = selfDividingNumbers(left, right)
# print(myList)



def diStringMatch(S):

	n = len(S) + 1
	l = list(permutations(range(0, n)))
	myList = []
	for i in l:
		if (testAssertion(S, i)):
			myList = i
			return myList

def testAssertion(S, myList):

	index = 0
	for i in range(0, len(myList)-1):
		try:
			if (S[index] == 'I'):
				assert (myList[i + 1] > myList[i])
			else:
				assert (myList[i + 1] < myList[i])
			index = index + 1
		except AssertionError:
			return False

	return True

#S = "DDI"
#S = diStringMatch(S)
#print(S)


def diStringMatch2(S):

	n = len(S)
	low = 0
	high = n
	result = []
	index = 0
	for i in S:
		if (i == 'I'):
			low = low + 1
			result[index] = low
		else:
			high = high - 1
			result[index] = high
		index = index + 1
	return result

# S = "DDI"
# S = diStringMatch2(S)
# print(S)

def kWeakestRows(mat, k):

	d = dict()
	for row in range(0, len(mat)):
		counter = 0
		i = 0
		while (i < len(mat[row][:]) and mat[row][i] == 1):
			i = i + 1
		d[row] = i

	od = OrderedDict(sorted(d.items(), key= lambda kv : (kv[1], kv[0])))
	result = []
	for key, v in od.items():
		result.append(key)
	return result[:k]

# k = 3
# mat = [[1,1,0,0,0],[1,1,1,1,0],[1,1,0,0,0],[1,0,0,0,0],[1,1,1,1,1]]
# result = kWeakestRows(mat, k)
# print(result)

def sortByBits(arr):

	arr.sort()
	d = dict()
	counter = 0
	for var in arr:
		counter = bin(var).count('1')
		d[var] = counter

	od = OrderedDict(sorted(d.items(), key=lambda kv: (kv[1], kv[0])))
	result = []
	for k, v in od.items():
		result.append(k)

	myIndex = 0
	i = 0
	while i < len(arr):
		elem = arr[i]
		counter = arr.count(elem)
		if (counter > 1):
			counter = counter - 1
			myIndex = result.index(elem, 0, len(result))
			for j in range(0, counter):
				result.insert(myIndex, elem)
			i = i + counter
		i = i + 1
	return result

# #arr = [10000,10000]
# arr = [0,1,2,3,4,5,6,7, 8, 8, 8]
# myList = sortByBits(arr)
# print(myList)

def reverseWords(s):

	myList = re.split(" ", s)
	result = str()
	for val in myList:
		stack = []
		for letter in val:
			stack.append(letter)
		while stack.__len__() > 0:
			result = result + stack.pop()
		result = result + ' '

	result = result.rstrip(' ')
	return result

# s= "Let's take LeetCode contest"
# result = reverseWords(s)
# print(result)

def reverseString(s):

	left = 0
	right = len(s)-1
	while right > left:

		tmp = s[left]
		s[left] = s[right]
		s[right] = tmp

		left = left + 1
		right = right - 1

	return s

# s = ["h","e","l","l","o"]
# s = reverseString(s)
# print(s)

def numberOfLines(widths, S):

	d = dict()
	myStr = "abcdefghijklmnopqrstuvwxyz"
	for i in range(0, len(myStr)):
		d[myStr[i]] = widths[i]

	line = 1
	result = []
	total = 0
	for k in S:

		value = d[k]
		if total + value <= 100:
			total = total + value
		else:
			total = value
			line = line + 1

	result.append(line)
	result.append(total)
	return result

# widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
# #widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
# S = "abcdefghijklmnopqrstuvwxyz"
# #S = "bbbcccdddaaa"
# myList = numberOfLines(widths, S)
# print(myList)

def findComplement(num):

	result = str()
	s = bin(num)
	flag = False
	for i in range(0, len(s)):
		if s[i] == 'b':
			flag = True
			continue

		if flag:
			if s[i] == '1':
				result = result + '0'
			else:
				result = result + '1'

	print(result)
	num = int(result, 2)
	return num

# num = findComplement(5)
# print(num)

def allCellsDistOrder(R, C, r0, c0):

	d = defaultdict(list)
	tmp = list()
	largest = 1000
	for i in range(0, R):
		for j in range(0, C):
			dist = abs(i - r0) + abs(j - c0)
			d[dist].append([i, j])

	od = OrderedDict(sorted(d.items(), key= lambda kv : kv[0]))
	result = []
	for k, v in od.items():
		#print(k, v)
		for elem in v:
			result.append(elem)
	return result

# R = 2
# C = 2
# r0 = 0
# c0 = 1
# myList = allCellsDistOrder(R, C, r0, c0)
# print2D(myList)

def factorial(n):
	if n <= 1:
		return n
	return n * max(n, factorial(n-1))

# val = factorial(8)
# print(val)

def shortestToChar(S: str, C: str):

	pos = []
	for i in range(0, len(S)):
		if S[i] == C:
			pos.append(i)

	print(pos)

	result = []
	for i in range(0, len(S)):
		tmpPos = len(S)
		for j in range(0, len(pos)):
			left = abs(i - pos[j])

			if left == 0:
				tmpPos = 0
				break

			tmpPos = min(tmpPos, left)
		result.append(tmpPos)
	return result

# S = "loveleetcode"
# C = 'e'
# myList = shortestToChar(S, C)
# print(myList)



# // add
# all
# indexes
# of
# TreeSet < Integer > set = new
# TreeSet <> ();

# C
# into
# tree
# set
# for (int i = 0;i < S.length();i++) if (S.charAt(i) == C) set.add(i);
#
# int[]
# result = new
# int[S.length()];
# for (int i = 0;i < S.length();i++){
# if (!set.contains(i)){
#
# Integer left = set.floor(i);
# Integer right = set.ceiling(i);
#
# if (left == null) left = Integer.MAX_VALUE;
# if (right == null) right = Integer.MAX_VALUE;
#
# result[i] = Math.min(Math.abs(left - i), Math.abs(right -i ));
#
# } else {
# result[i] = 0;
# }
# }
# return result;



def calPoints(s):

	res = []
	for i in range(0, len(s)):

		try:
			tmp = int(s[i])
			res.append(tmp)
		except ValueError:
			if s[i] == "+":
				if len(res) >= 2:
					tmp = res[-1] + res[-2]
					res.append(tmp)

			elif s[i] == "C":
				if len(res) >= 1:
					res.pop(-1)

			elif s[i] == "D":
				if len(res) >= 1:
					res.append(2 * res[-1])
	return sum(res)

# myList = ["5","-2","4","C","D","9","+","+"]
# val = calPoints(myList)
# print(val)

def nextGreaterElement(nums1, nums2):

	res = []
	print(nums2)

	for num in nums1:
		flag = False
		for i in range(0, len(nums2)):

			if num == nums2[i]:
				flag = True
			elif flag and nums2[i] > num:
				res.append(nums2[i])
				flag = False
				break

		if flag and i == len(nums2)-1:
			res.append(-1)

	return res



nums1 = [3,1,5,7,9,2,6]
nums2 = [1,2,3,5,6,7,9,11]

#nums1 = [2,4]
#nums1 = [4, 1, 2]
#nums2 = [1, 3, 4, 2]
#nums2 = [1,2,3,4]
# myList = nextGreaterElement(nums1, nums2)
# print(myList)

# for k in s:
# 	if num > k:
# 		flag = True
# 		res.append(k)
# 		break

def maxPower(s):

	power = 0
	i = 0

	while i < len(s)-1:
		count = 1
		while i < len(s)-1 and s[i] == s[i+1]:
			count = count + 1
			i = i + 1
		else:
			i = i + 1
		power = max(power, count)

	return power

# s = "tourist"
# val = maxPower(s)
# print(val)

def transpose(A):
	return [list(row) for row in zip(*A)]

# myList = [[1, 2, 3, 5], [4, 5, 6, 8], [7, 8, 9, 10]]
# myList = transpose(myList)
# print2D(myList)

def isToeplitzMatrix(matrix):

	row = len(matrix)
	column = len(matrix[0])

	result = True
	for i in range(0, row - 1):
		for j in range(0, column):
			if j > 0:
				if matrix[i][j-1] != matrix[i + 1][j]:
					return False

	return True
# matrix = [[1,2,3,4], [5,1,2,3], [9,5,1,2]]
# result = isToeplitzMatrix(matrix)
# print(result)

# def isPrime(num):
#
# 	if num > 1:
# 		# check for factors
# 		for i in range(2, num):
# 			if (num % i) == 0:
# 				return False
# 		else:
# 			return True
# 	else:
# 		return False


def countPrimeSetBits(L, R):

	setCount = 0
	for i in range(L, R):
		count = 0
		s = bin(i)
		for a in s:
			if a == '1':
				count = count + 1

		if count > 1:
			isPrime = True
			for j in range(2, count):
				if count % j == 0:
					isPrime = False
					break

			if isPrime == True:
				setCount = setCount + 1

	return setCount

# L = 244
# R = 269
# val = countPrimeSetBits(L, R)
# print(val)

def canConstruct(ransomNote, magazine):


	cnt = Counter()
	for letter in magazine:
		cnt[letter] += 1

	for a in ransomNote:
		if not cnt.__contains__(a):
			return False
		elif cnt[a] == 0:
			return False
		else:
			cnt[a] -= 1
	return True

# ransomNote = "aa"
# magazine = "aab"
# val = canConstruct(ransomNote, magazine)
# print(val)


def guessNumber(n):

	random.seed(None, version=2)
	low = 1
	high = n
	result = 1
	while not result == 0:

		sample = random.randint(low, high)
		print(sample)
		if result == 1:
			if sample > 1:
				high = sample - 1
		elif result == -1:
			if sample < 10:
				low = sample + 1

# n = 10
# sample = guessNumber(n)
# print(sample)


def isPowerOfFour(num):

	if num < 1:
		return False

	result = math.log(num, 4)
	s = str(result)
	if s[len(s) - 1] == '0':
		return True
	return False

# num = 16
# result = isPowerOfFour(num)
# print(result)


def isPowerOfThree(n):

	if n < 1:
		return False

	result = math.log(n, 3)
	res_ceil = math.ceil(result)
	if res_ceil - result < 0.00001:
		print(result)
		print(res_ceil)
		return True
	return False

# num = 19682
# result = isPowerOfThree(num)
# print(result)


class MyStack:

	import queue
	def __init__(self):
		""""
		Initialize your data structure here.
		"""
		q = queue.LifoQueue()


	def push(self, x):
		"""
		Push element x onto stack.
		"""
		q.put(x)

	def pop(self):
		"""
		Removes the element on top of the stack and returns that element.
		"""
		q.get()

	def top(self):
		"""
		Get the top element.
		"""

	def empty(self):
		"""
		Returns whether the stack is empty.
		"""


def fizzBuzz(n):

	myList = []
	for i in range(1, n + 1):

		if (i % 3 == 0) and (i % 5 == 0):
			myList.append("FizzBuzz")
		elif (i % 3 == 0):
			myList.append("Fizz")
		elif (i % 5 == 0):
			myList.append("Buzz")
		else:
			myList.append(str(i))
	return myList

# n = 15
# result = fizzBuzz(n)
# print(result)

def countSegments(s):

	s = s.lstrip(" ")
	s = s.rstrip(" ")
	if len(s) == 0:
		return 0

	myList = []
	myList = s.split(' ')
	count = 0
	for a in myList:
		if a != "":
			count = count + 1
	return count

# s = ", , , ,        a, eaefa"
# val = countSegments(s)
# print(val)

def isSubsequence(s, t):

	set_s = set(s)
	set_t = set(t)

	if not set_s.issubset(set_t):
		return False

	start = 0
	end = len(t)
	j = 0
	newStr = str()
	for a in s:
		pos = t.find(a, start, end)
		if pos == -1:
			return False
		newStr = newStr + a
		start = pos + 1

	return newStr == s
	# d1 = defaultdict(list)
	# for i in range(0, len(s)):
	# 	d1[s[i]].append(i)
	#
	# d2 = defaultdict(list)
	# j = 0
	# for i in range(0, len(t)):
	# 	if set_s.__contains__(t[i]):
	# 		d2[t[i]].append(j)
	# 		j = j + 1
	#
	# return d1 == d2

# s = "aabc"
# t = "ahabgdc"
# result = isSubsequence(s, t)
# print(result)

def repeatedSubstringPattern(s):

	if len(s) < 2:
		return False

	set_s = set(s)
	if len(set_s) == 1:
		return True

	startChar = s[0]
	i = len(s) - 1
	while i > 0 and s[i] != startChar:
		i = i - 1

	###No pattern found
	if i == len(s)-1:
		return False

	while i > 0 and s[i] == startChar:
		i = i - 1

	startPos = i + 1

	#find the pattern
	pattern = []
	i = 0
	while i < len(s) and startPos < len(s) and s[i] == s[startPos]:

		pattern.append(s[i])
		i = i + 1
		startPos = startPos + 1

	print(len(pattern))

	if len(pattern) <= 1:
		return False


	start = 0
	end = start + len(pattern)

	while end < len(s):
		if list(s[start: end]) != pattern:
			print(" here not there ")
			return False
		start = start + len(pattern)
		end = start + len(pattern)

	print(start)
	print(end)
	if end < len(s):
		return False
	else:
		return True

# #Wrong code
# s = "abaababaab"
# val = repeatedSubstringPattern(s)
# print(val)

def diStringMatch(S):

	low = 0
	high = len(S)
	res = []
	for i in range(0, len(S)):
		if S[i] == 'I':
			res.append(low)
			low += 1
		elif S[i] == 'D':
			res.append(high)
			high -= 1

	if S[len(S)-1] == 'I':
		res.append(high)
	else:
		res.append(low)

	return res

# S = "IIID"
# s = diStringMatch(S)
# print(s)

# def reorderLogFiles(logs):
#
# 	tmp = []
# 	nums = []
# 	for a in logs:
# 		tmp = a.split(' ')
# 		if tmp[1].isdecimal():
# 			nums.append(a)
# 			logs.remove(a)
#
# 	d = dict()
# 	tmp = []
# 	for a in logs:
# 		tmp = a.split(' ')
# 		d[tmp[0]] = tmp[1]
#
# 	od = OrderedDict(sorted(d.items(), key=lambda kv: [kv[1], kv[0]]))
# 	res = []
# 	for k in od.keys():
# 		print(k)
#
#
# 	return logs
#
# logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
# #output = reorderLogFiles(logs)
# #print(output)

# tmp = []
# logs = sorted(logs)
# print(logs)

# myList = list(tuple())
# for a in logs:
# 	tmp = a.split()
# 	tmp.append(a[0])
# 	tmp.append(a[1])
# 	myList.append(tmp)
#
#
# for k in myList:
# 	print(k[0], k[1])
#
#
# logs.sort((key=lambda: myList[1], myList[0]))
# print(logs)

# logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
#
# # def haveNum(s):
# # 	tmp = []
# # 	tmp = s.split(' ')
# # 	return tmp[1].isdecmial()
#
# x = [i for i in range(0, 10)]
# #y = [i**2 for i in x]
# y = map(lambda i: i**2, x)
# z = list(y)
# d =
#
# #map(lambda i: pow(i, 2), x)
# print(z)
#


# def findDifferenceBy2(myList):
#
# 	result = []
# 	for row in myList:
# 		tmp = []
# 		row.sort()
# 		print(row)
# 		for i in range(0, len(row)):
# 			if row[i+1] - row[i] == 2:
# 				tmp.append(row[i])
# 				tmp.append(row[i+1])
# 				if len(tmp) == 2:
# 					print(tmp)
# 					result.append(tmp)
# 					tmp.clear()
# 	return result
#
# myList = [[1, 2, 3, 4], [4, 1, 2 , 3], [1, 23, 3, 4, 7]]
# result = findDifferenceBy2(myList)
# print(result)

def rotateString(A, B):

	l = len(A)
	for i in range(0, l-1):
		letter = A[-1]
		A = A[:-1]
		A = letter + A

		if A == B:
			return True
	return False

# A = 'abcde'
# # B = 'cdeab'
# # result = rotateString(A, B)
# # print(result)

def bin2Decimal(x):

	value = 0
	x = x[::-1]
	print(x)

	for i in range(0, len(x)):
	 	value = value + int(x[i]) * pow(2, i)
	return value

def rotatedDigits(N):

	d = dict()
	d['6'] = True
	d['9'] = True
	d['2'] = True
	d['5'] = True
	counter = 0

	for i in range(1, N + 1):
		tmp = list(str(i))
		for j in tmp:
			if d.get(j):
				counter = counter + 1
	return counter


#new = [list(map(lambda x: d[x],  list(str(i)))) for i in range(1, N + 1)]



# N = 857
# counter = rotatedDigits(N)
# print(counter)

#N = 3
#rotatedDigits(N)
# #nums = [i if i > 2 else i + 10 for i in range(1, N + 1)]
# nums = [i for i in range(1, N + 1)]
# evenNums = list(filter(lambda x: x % 2 == 0, nums))
# print(evenNums)
#
# oddNums = list(filter(lambda x: x % 2 == 1, nums))
# print(oddNums)
#
# nV = list(filter(lambda x: x not in ['a', 'i', 'e', 'o', 'u', ' '], "mina shenouda"))
# print(nV)
#
# squares = list(map(lambda x: x**2, oddNums))
# print(squares)
#
# A = "MINA SHENOUDA"
# A = A[2::-1] + A[:2:-1]
# print(A)
#
# nums = nums[2::-1] + nums[:2:-1]
# print(nums)

# N = 10
# nums = list(range(1, N+1))
# strsRotated = list(map(lambda x: bin(x).replace("0b", "")[::-1], nums))
# numsRotated = list(map(lambda x: bin2Decimal(x), strsRotated))
# print(numsRotated)

def busyStudent(startTime, endTime, queryTime):

	nums = list(zip(startTime, endTime))
	result = list(filter(lambda x: x[0] <= queryTime and queryTime <= x[1], nums))
	print(len(result))


# startTime = [1, 2, 3, 4]
# endTime = [3, 2, 7, 9]
# queryTime = 4
#startTime = [9,8,7,6,5,4,3,2,1]
#endTime = [10,10,10,10,10,10,10,10,10]
#queryTime = 5
# result = busyStudent(startTime, endTime, queryTime)
# print(result)

def maxScore(s):

	maxCounter = -1
	for i in range(1, len(s)):
		counter_0 = reduce(lambda counter, x: counter + 1 if x == '0' else counter, s[:i], 0)
		counter_1 = reduce(lambda counter, x: counter + 1 if x == '1' else counter, s[i:], 0)
		maxCounter = max(maxCounter, counter_0 + counter_1)
	return maxCounter

#my_numbers = [1, 2, 3, 4, 5]
# Fix all three respectively.
#reduce_result = reduce(lambda num1, num2: num1 * num2, my_numbers, 1)
#print(reduce_result)
# s = "011101"
# value = maxScore(s)
# print(value)


def reformat(s):

	nums = list(filter(lambda x: str.isdigit(x), s))
	chars = list(filter(lambda x: str.isalpha(x), s))

	result = str()
	if abs(len(nums) - len(chars)) > 1:
		return result

	longer = nums if len(nums) >= len(chars) else chars
	shorter = chars if len(nums) >= len(chars) else nums

	if len(longer) > len(shorter):
		result = longer[0]
		longer.remove(longer[0])

	for i in range(0, len(longer)):
		result = result + shorter[i]
		result = result + longer[i]

	return result

# s = "a0b1c2"
# s = reformat(s)
# print(s)


def stringMatching(words):

	words.sort(key=lambda x: len(x))
	result = set()
	for a in words:
		for b in words:
			if a.find(b) > -1 and a != b:
				result.add(b)
	return list(result)

# words = ["leetcoder","leetcode","od","hamlet","am"]
# result = stringMatching(words)
# print(result)



def countLargestGroup(n):

	d = dict()
	largest = -1
	for i in range(1, n+1):
		if i < 10:
			d[i] = 1
		else:
			value = 0
			value = reduce(lambda value, x: value + int(x), str(i), 0)
			if d.__contains__(value):
				d[value] += 1
			else:
				d[value] = 1

	largest = max(d.items(), key= lambda kv: kv[1])
	d = sorted(d.items(), key= lambda x: x[1], reverse=True)
	counter = reduce(lambda counter, x: counter + 1 if x[1] == largest[1] else counter, d, 0)
	return counter

# n = 13
# val = countLargestGroup(n)
# print(val)

def isZero(n):

	s = str(n)
	for i in range(0, len(s)):
		if s[i] == '0':
			return False
	return True

def getNoZeroIntegers(n):

	result = []
	first = 0
	last = n
	while len(result) < 2 :

		first = first + 1
		last = last - 1
		print(first)
		print(last)

		if not isZero(first):
			continue

		if not isZero(last):
			continue

		if first + last != n:
			continue

		result.append(first)
		result.append(last)

	return result

# n = 2
# result = getNoZeroIntegers(n)
# print(result)

def tictactoe(moves):

	import numpy as np
	tmp = ["" for i in range(0, 9)]
	grid = np.array(tmp).reshape(3, 3)

	turn = 0
	for move in moves:
		row = move[0]
		column = move[1]
		if grid[row][column] == "":
			if turn % 2 == 0:
				grid[row][column] = 'X'
			else:
				grid[row][column] = 'O'

			if turn >= 3:
				for i in range(0, 3):

					if all(grid[i, :] == 'X') or all(grid[i, :] == 'O'):
						return "A" if turn % 2 == 0 else "B"

					if all(grid[:, i] == 'X') or all(grid[:, i] == 'O'):
						return "A" if turn % 2 == 0 else "B"

				if all(np.diagonal(grid, offset=0) == 'X') or all(np.diagonal(grid, offset=0) == 'O'):
					return "A" if turn % 2 == 0 else "B"

				if all(np.fliplr(grid).diagonal() == 'X') or all(np.fliplr(grid).diagonal() == 'O'):
					return "A" if turn % 2 == 0 else "B"
		turn += 1

	for i in range(0, 3):
		if any(grid[i, :] == ""):
			return "Pending"

	return "Draw"

#moves = [[0,0],[2,0],[1,1],[2,1],[2,2]]
#moves = [[0,0],[2,0]]
#moves = [[0,0],[1,1],[2,0],[1,0],[1,2],[2,1],[0,1],[0,2],[2,2]]
# result = tictactoe(moves)
# print(result)

def minTimeToVisitAllPoints(points):

	prev = [0, 0]
	time = 0

	for point in points:
		d1 = abs(point[0] - prev[0])
		d2 = abs(point[1] - prev[1])
		print(d1, d2)
		if all(prev) != 0:
			if d1 == d2:
				time = time + d1
			else:
				smaller = min(d1, d2)
				time += smaller
				time += abs(point[1] - point[0])
		prev = point
	return time

#points = [[1,1],[3,4],[-1,0]]

# points= [[3,2],[-2,2]]
# result = minTimeToVisitAllPoints(points)
# print(result)

def isLeapYear(year):

	if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
		return True
	return False

def dayCountByDate(year1, month1, day1):

	leapCount = 0
	daysCount = 0

	for i in range(1972, year1, 4):
		if isLeapYear(i):
			leapCount += 1

	if year1 > 1971:
		daysCount = (year1 - leapCount - 1971) * 365 + leapCount * 366

	d = dict()
	d[1] = 31
	d[2] = 28
	d[3] = 31
	d[4] = 30
	d[5] = 31
	d[6] = 30
	d[7] = 31
	d[8] = 31
	d[9] = 30
	d[10] = 31
	d[11] = 30
	d[12] = 31

	for i in range(1, month1):
		daysCount = daysCount + d[i]

	daysCount = daysCount + day1

	if isLeapYear(year1) and month1 > 2:
		daysCount += 1

	return daysCount

def daysBetweenDates(date1, date2):

	year1, month1, day1 = date1.split('-')
	year2, month2, day2 = date2.split('-')

	year1, month1, day1 = int(year1), int(month1), int(day1)
	year2, month2, day2 = int(year2), int(month2), int(day2)

	dayCount1 = dayCountByDate(year1, month1, day1)
	dayCount2 = dayCountByDate(year2, month2, day2)
	return abs(dayCount1 - dayCount2)

#date1 = "2019-06-29"
#date2 = "2019-06-30"
#date1 = "2020-01-15"
#date2 = "2019-12-31"

# date1 = "1971-06-29"
# date2 = "2010-09-23"
# result = daysBetweenDates(date1, date2)
# print(result)


def numSmallerHelper(words):

	sizes = []
	for word in words:
		od = OrderedDict()
		cnt = Counter(word)
		od = OrderedDict(sorted(cnt.items(), key=lambda kv: kv[0]))
		for k, v in od.items():
			sizes.append(v)
			break

	return sizes

def numSmallerByFrequency(queries, words):

	wordSizes = numSmallerHelper(words)
	wordSizes.sort()

	querySizes = numSmallerHelper(queries)
	result = []
	for query in querySizes:
		tmp = list(filter(lambda x: x > query, wordSizes))
		result.append(len(tmp))

	return result

# queries = ["bba","abaaaaaa","aaaaaa","bbabbabaab","aba","aa","baab","bbbbbb","aab","bbabbaabb"]
# words = ["aaabbb","aab","babbab","babbbb","b","bbbbbbbbab","a","bbbbbbbbbb","baaabbaab","aa"]
# result = numSmallerByFrequency(queries, words)
# print(result)

def dayOfYear(date):

	year, month, day = date.split('-')
	year, month, day = int(year), int(month), int(day)
	daysCount = 0

	d = dict()
	d[1] = 31
	d[2] = 28
	d[3] = 31
	d[4] = 30
	d[5] = 31
	d[6] = 30
	d[7] = 31
	d[8] = 31
	d[9] = 30
	d[10] = 31
	d[11] = 30
	d[12] = 31

	for i in range(1, month):
		daysCount = daysCount + d[i]

	daysCount = daysCount + day
	if isLeapYear(year) and month > 2:
		daysCount += 1

	return daysCount

# date = "2019-01-09"
# result = dayOfYear(date)
# print(result)

def numPairsDivisibleBy60(time):

	counter = 0
	repeat = 0
	cnt = Counter(time)
	od = OrderedDict(sorted(cnt.items(), key=lambda kv: kv[0]))
	# for k, v in od.items():
	# 	print(k , v)
	for k, v in od.items():
		if v > 1 and (2 * k) % 60 == 0:
			repeat = v - 1
			while repeat > 0:
				counter += repeat
				repeat = repeat - 1
		for k2, v2 in od.items():
			if k2 > k and (k + k2) % 60 == 0:
				print(k, k2)
				counter += v * v2
	return counter

# time = [30,20,150,100,40]
# time = [60, 60, 60]
# result = numPairsDivisibleBy60(time)
# print(result)

def largestSumAfterKNegations(A, K):

	if len(A) == 1:
		return abs(A[0]) if K % 2 == 0 else -abs(A[0])

	A.sort()
	print(A)

	if K == 1:
		result = -A[0]
		result = result  + reduce(lambda counter, x: counter + x, A[1:], 0)
	elif A[0] >= 0:
	    result = reduce(lambda counter, x: counter + x, A, 0)
	elif A[0] < 0 and K % 2 == 0:
		result = abs(A[0]) + -A[1]
		result = result + reduce(lambda counter, x: counter + x, A[2:], 0)
	elif A[0] < 0 and K % 2 == 1:
		result = abs(A[0])
		result = result + reduce(lambda counter, x: counter + x, A[1:], 0)
	return result

# #A = [2, -3, -1, 5, -4]
# #A = [4,2,3]
# A = [-2,5,0,2,-2]
# K = 3
# result = largestSumAfterKNegations(A, K)
# print(result)

def detectCapitalUse(word):
	return word.isupper() or word.istitle() or word.islower()

# #word = "FlaG"
# word = "USA"
# result = detectCapitalUse(word)
# print(result)

def numRookCaptures(board):

	import numpy as np
	grid = np.array(board)
	row, column = grid.shape
	pos = tuple()

	flag = False
	for i in range(0, row):
		for j in range(0, column):
			if grid[i, j] == "R":
				pos = i, j
				flag = True
				break
		if flag:
			break

	pawn = 0
	r, c = pos
	for j in range(c+1, column):
		if grid[r, j] == 'B':
			break

		if grid[r, j] == 'p':
			pawn += 1
			break

	for j in range(c-1, 0, -1):
		if grid[r, j] == 'B':
			break

		if grid[r, j] == 'p':
			pawn += 1
			break

	for i in range(r+1, row):
		if grid[i, c] == 'B':
			break

		if grid[i, c] == 'p':
			pawn += 1
			break

	for i in range(r-1, 0, -1):
		if grid[i, c] == 'B':
			break

		if grid[i, c] == 'p':
			pawn += 1
			break

	return pawn

# board = [[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","p",".",".",".","."],["p","p",".","R",".","p","B","."],[".",".",".",".",".",".",".","."],[".",".",".","B",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."]]
# # result = numRookCaptures(board)
# # print(result)

def findRelativeRanks(nums):

	import numpy as np
	nums = np.array(nums)
	result = ["" for i in range(0, len(nums))]

	min = -1000
	counter = 0
	for i in range(0, len(nums)):
		args = np.argmax(nums)
		nums[args] = min
		if counter == 0:
			result[args] = "Gold Medal"
		elif counter == 1:
			result[args] = "Silver Medal"
		elif counter == 2:
			result[args] = "Bronze Medal"
			counter += 1
		else:
			result[args] = str(counter)
		counter += 1
	return result

# nums = [10,3,8,9,4]
# result = findRelativeRanks(nums)
# print(result)


def lemonadeChange(bills):

	d = {5: 0, 10: 0, 20: 0}
	result = True
	for bill in bills:
		d[bill] = d[bill] + 1
		if bill > 5:
			rem = bill - 5
			while rem > 0:
				if rem > 20 and d[20] > 0:
					rem = rem - 20
					d[20] = d[20] - 1
				elif rem > 10 and d[10] > 0:
					rem = rem - 10
					d[10] = d[10] - 1
				elif d[5] > 0:
					rem = rem - 5
					d[5] = d[5] - 1
				else:
					return False
	return True



# bills = [5, 5, 5, 10, 20]
# bills = [5,5,10]
# bills =  [5,5,10,10,20]
# bills = [5,5,5,10,5,5,10,20,20,20]
# result = lemonadeChange(bills)
# print(result)

def findSpecialInteger(arr):

	d = dict()
	for val in arr:
		if d.__contains__(val):
			d[val] += 1
		else:
			d[val] = 1

	size = 0.25*len(arr)
	result = list(filter(lambda kv: kv[1] > size, d.items()))
	return result[0][0]


# arr = [1, 2, 2, 6, 6, 6, 6, 7, 10]
# # result = findSpecialInteger(arr)
# # print(result)

def minDeletionSize(A):

	result = 0
	for col in zip(*A):
		if any(col[i] > col[i + 1] for i in range(len(col) - 1)):
			result += 1
	return result

#A = ["cba","daf","ghi"]
# A = ["zyx", "wvu", "tsr"]
# #A = ["a","b"]
# result = minDeletionSize(A)
# print(result)

def minStartValue(nums):

	index = 0
	start = 0
	while True:

		result = 0
		start = start + 1
		for index in range(0, len(nums)):

			if index == 0:
				result += start + nums[index]
			else:
				result = result + nums[index]

			if result < 1:
				break

		if result > 0 and index == len(nums) - 1:
			break

	return start

#nums = [-3,2,-3,4,2]
# nums = [1,-2,-3]
# result = minStartValue(nums)
# print(result)

def smallestRangeI(A, K):

	A.sort()
	elem = [i for i in range(-K, K + 1)]
	D0 = map(lambda x: abs(x - A[0]), elem)
	D1 = map(lambda x: abs(x - A[-1]), elem)

	result = 10000
	for i in D0:
		for j in D1:
			result = min(result, abs(i - j))

	return result

# A = [0,10]
# A = [1,3,6]
# #K = 2
# K = 3
# result = smallestRangeI(A, K)
# print(result)

def peakIndexInMountainArray(A):

	for i in range(len(A)):
		if A[i] > A[i + 1]:
			return i

# A = [0,1,0]
# result = peakIndexInMountainArray(A)
# print(result)

def isPrefixOfWord(sentence, searchWord):

	result = -1
	words = sentence.split(' ')
	l = len(searchWord)
	for i in range(len(words)):
		if searchWord == words[i][0:l]:
			return i + 1
	return result

# sentence = "i love eating burger"
# searchWord = "burg"
# result = isPrefixOfWord(sentence, searchWord)
# print(result)

def projectionArea(grid):
	pass
#
# def numSpecialEquivGroups(A):
#
# 	groups = []
# 	for word in A:
# 		evens = []
# 		odds = []
# 		[evens.append(word[i]) if i % 2 == 0 else odds.append(word[i]) for i in range(len(word))]
# 		evens.sort()
# 		odds.sort()
# 		evens.extend(odds)
# 		groups.append(evens)
#
#
# 	groups.sort()
# 	i = 0
# 	while i < len(groups)-1:
# 		if groups[i] == groups[i + 1]:
# 			groups.pop(i)
# 		else:
# 			i = i + 1
# 	return len(groups)

def numSpecialEquivGroups(A):
        def count(A):
            ans = [0] * 52
            for i, letter in enumerate(A):
                ans[ord(letter) - ord('a') + 26 * (i%2)] += 1
            return tuple(ans)
        return len({count(word) for word in A})



# A = ["abcd", "cdab", "cbad", "xyzz", "zzxy", "zzyx"]
# #A = ["abc","acb","bac","bca","cab","cba"]
# result = numSpecialEquivGroups(A)
# print(result)

def binaryGap(N):

	s = bin(N)
	result =0
	i = 0
	prev = -1
	while i < len(s):
		if s[i] == '1':
			if prev == -1:
				prev = i
			else:
				result = max(result, abs(i - prev))
				prev = i

		i = i + 1

	return result

# N = 22
# result = binaryGap(N)
# print(result)

def matrixReshape(nums, r, c):
	import numpy as np

	size = np.shape(nums)
	if r * c != size[0] * size[1]:
		return nums
	nums = np.reshape(nums, (r, c))
	return nums


# nums = [[1, 2],[3, 4]]
# r = 1
# c = 4
# result = matrixReshape(nums, r, c)
# print(result)

def distributeCandies(candies, num_people):

	result = [0 for i in range(num_people)]
	index = 0
	candy = 0
	while candies > 0:
		if candies > candy:
			candy += 1
		else:
			candy = candies

		result[index] = result[index] + candy
		candies = candies - candy
		index = (index + 1) % num_people

	return result

# candies = 7
# num_people = 4
# result = distributeCandies(candies, num_people)
# print(result)

def rotate(l, n):
	return l[n:] + l[:n]


def shiftGrid(grid, k):

	import numpy as np
	vec = []
	for elem in grid:
		vec.extend(elem)

	size = np.shape(grid)
	for i in range(k):
		vec = vec[-1:] + vec[:-1]
	grid = np.reshape(vec, (size[0], size[1]))
	return grid

# grid = [[1,2,3],[4,5,6],[7,8,9]]
# grid  = [[1],[2],[3],[4],[7],[6],[5]]
# k = 23
# result = shiftGrid(grid, k)
# print(result)

def fairCandySwap(A, B):

	A.sort(reverse=True)
	B.sort(reverse=True)

	zipped = list(zip(A, B))
	print(A)
	print(B)
	print(zipped)
	for elem in zipped:
		if (sum(A) > sum(B) and elem[0] > elem[1]) or (sum(A) < sum(B) and elem[0] < elem[1]):
			result = [elem[0], elem[1]]
			return result

	for i in A:
		for j in B:
			if (sum(A) > sum(B) and i > j) or (sum(B) > sum(A) and i < j):
				result = [i, j]
				return result

	result = [A[0], B[0]]
	return result

#A = [1, 1]
#B = [2, 2]
#A = [1,2,5]
#B = [2,4]
#A = [35,17,4,24,10]
#B = [63,21]
# A = [32,38,8,1,9]
# B = [68,92]

# print(sum(A))
# print(sum(B))
#print(tmpList)

#A = [20,35,22,6,13]
#B = [31,57]
# result = fairCandySwap(A, B)
# print(result)

def reverseOnlyLetters(S):

	result = list(filter(lambda x: x.isalpha(), S))
	result.reverse()

	for i in range(len(S)):
		if not S[i].isalpha():
			result.insert(i, S[i])
	S = "".join(result)
	return S

# #S = "ab-cd"
# S = "a-bC-dEf-ghIj"
# result = reverseOnlyLetters(S)
# print(result)

def arrayRankTransform2(arr):

	import numpy as np
	minval = -1
	result = [0 for i in range(len(arr))]
	rank = 0
	for i in range(len(arr)):
		ind = np.unravel_index(np.argmin(arr), np.shape(arr))[0]
		result[ind] = rank

		if minval != arr[ind]:
			rank += 1

		minval = arr[ind]
		arr[ind] = 100000
		result[ind] = rank
	return result

def arrayRankTransform(arr):

	d = defaultdict(list)
	for i in range(len(arr)):
		d[arr[i]].append(i)

	for k, v in d.items():
		print(k, v)

	od = OrderedDict(sorted(d.items(), key=lambda kv: kv[0]))
	result = [0 for i in range(len(arr))]
	rank = 1
	for k, v in od.items():
		for pos in v:
			result[pos] = rank
		rank += 1
	return result

# arr = [37,12,28,9,100,56,80,5,12]
# #arr = [40, 10, 20, 30]
# #arr = [100, 100, 100]
# result = arrayRankTransform(arr)
# print(result)

def largestPerimeter(A):

	def checkTriangle(a, b, c):
		result = True
		if a + b <= c or a + c <= b or b + c <= a:
			result = False
		return result

	perimeter = 0
	A.sort()
	for i in range(len(A)):
		tri = A[i:i+3]
		if len(tri) == 3 and checkTriangle(tri[0], tri[1], tri[2]):
			perimeter = max(perimeter, sum(tri))
	return perimeter
# A = [1,1,2]
# #A = [3,2,3,4]
# result = largestPerimeter(A)
# print(result)

def removePalindromeSub(s):

	def isPalindrome(s):
		mid = len(s)
		last = -1
		for i in range(mid):
			if s[i] != s[last]:
				return False
			last -= 1
		return True





	k = 0
	counter = 0
	l = list(s)
	n = len(l)
	l[:(n - k)] = ""
	print(l)

	while all(l) != "":
		tmp = l[:k]
		if isPalindrome(tmp):
			counter += 1
			l[:(n - k)] = ""
			k = 0
			n = len(l)
		else:
			k += 1

	return counter

# s = "ababa"
# result = removePalindromeSub(s)
# print(result)

def hasAlternatingBits(n):

	s = bin(n)
	s = s[2:]
	pos = s.find('1', 0, len(s))
	if pos == -1:
		return False
	ones = [i if s[i] == '1' else -1 for i in range(len(s))]
	prev = -1
	for i in range(len(ones)):
		if ones[i] > -1:
			if prev > -1 and i - prev != 2:
				return False
			prev = i
			if prev + 2 < len(ones) and ones[prev + 2] < 0:
				return False
	return True

# n = 2
# result = hasAlternatingBits(n)
# print(result)

def countBinarySubstrings(s):
	pass


def shortestCompletingWord(licensePlate, words):

	licensePlate = licensePlate.lower()
	cntPlate = Counter(licensePlate)

	shortest = 1000
	d = dict()
	for i in range(len(words)):
		cntWord = Counter(words[i])

		platedist = 0
		worddist = 0
		for k, v in cntWord.items():
			if cntPlate.__contains__(k):
				minval = min(v, cntPlate[k])
				platedist = platedist + minval
				worddist = worddist + minval

		d[i] = abs(len(licensePlate) - platedist - len(words[i]) + worddist)
		print(d[i])

	od = OrderedDict(sorted(d.items(), key=lambda kv: (kv[1], kv[0])))
	for k, v in od.items():
		return words[k]


licensePlate = "1s3 PSt"
words = ["step", "steps", "stripe", "stepple"]
#licensePlate ="1s3 456"
#words = ["looks","pest","stew","show"]
result = shortestCompletingWord(licensePlate, words)
print(result)

# if cntPlate[k] > v:
# 	cntWord[k] = cntPlate[k] - v
# else:
# 	cntWord[k] = v - cntPlate[k]
