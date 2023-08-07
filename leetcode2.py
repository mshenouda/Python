import numpy as np
from collections import OrderedDict
from operator import itemgetter
import re
from collections import Counter

def returnNum(tup):
	print(tup)
	return tup


# od.sort(itemgetter(1))
# # print(od)
# for k, v in d.items():
# 	print(k, v)

def intToRoman(num):
	d = {1000: 'M', 500: 'D', 900: 'CM', 400: 'CD', 100: 'C', 90: 'XC', 40: 'XL', 50: 'L', 9: 'IX', 10: 'X', 4: 'IV',
	     5: 'V', 1: 'I'}
	d = dict(sorted(d.items(), key=lambda item: item[0], reverse=True))
	result = str()
	while num > 0:
		for k, v in d.items():
			if num >= k:
				result = result + v
				num = num - k
				print(num)
				break
	return result


def isEmpty(l):
	return len(l) == 0


def to_1_9(l):
	a = range(10)
	a = np.array(a, dtype=str)
	for i in l:
		if i not in a:
			return False
	return True


def isduplicate(l):
	return len(np.unique(l)) < len(l)


def shiftingLetters(S, shifts):
	# from collections import deque
	# S = list(S)
	# letters = [chr(ord('a') + i) for i in range(26)]
	# letters = deque(letters)
	# for i in range(len(shifts)):
	# 	for j in range(i+1):
	# 		letters.rotate(shifts[i])
	# 		print(letters)
	# # print(x)            # [3, 4, 0, 1, 2]

	# S = "".join(S)
	# return S
	S = list(S)
	for i in range(len(shifts)):
		for j in range(i + 1):
			val = ord(S[j]) + shifts[i] % 26
			if val > ord('z'):
				val = ord('a') + val - ord('z') - 1
			S[j] = chr(val)

	S = "".join(S)
	return S


def convert(l):
	l = str(l[0]) + '->' + str(l[-1])
	return l


def summaryRanges(l):
	l.sort()
	ans = []
	result = []

	# for i in l:
	# 	if (ans != []) & (i == ans[-1] + 1):
	# 		ans.append(i)
	# 	elif len(ans) > 0:
	# 		s = convert(ans)
	# 		result.append(s)
	# 	else:
	# 		ans = [].append(i)
	#
	# s = convert(ans)
	# result.append(s)
	# return result
	result = []
	start = finish = l[0]
	for i in range(1, len(l)):
		if l[i] == (l[i - 1] + 1):
			finish = l[i]
		else:
			if i == len(l) - 1:
				result.append(str(start) + '->' + str(l[i]))
			else:
				result.append(str(start) + '->' + str(l[i - 1]))
				start = finish = l[i]
	return result


class NumArray:

	def __init__(self, nums):
		self.nums = nums

	def sumRange(self, i, j):
		result = np.sum(self.nums[i:j + 1])
		return result


def trailingZeroes(n):
	def zero_count(v):
		v = str(v)
		w = v.strip('0')
		return len(v) - len(w)

	count = 0
	a = list(range(1, n + 1))
	for i in a:
		if i % 5 == 0:
			count += zero_count(i * (i - 1))
	return count


def remove_(s):
	s = list(s)
	i = 0
	for i in range(len(s) - 1):
		if (s[i] != '#' or s[i] != '') and (s[i + 1] == '#'):
			s[i] = ''
			s[i + 1] = ''
			s = list(filter(lambda x: x != '', s))
			s = remove_(s)
			break
	return "".join(s)


def backspaceCompare(S: str, T: str) -> bool:
	def remove_f(s: str) -> str:
		s = list(s)
		i = 0
		for i in range(len(s) - 1):
			if (s[i] != '#' or s[i] != '') and (s[i + 1] == '#'):
				s[i] = ''
				s[i + 1] = ''
				s = list(filter(lambda x: x != '', s))
				s = remove_f(s)
				break
		return "".join(s)

	s1 = remove_(S)
	s2 = remove_(T)
	s1 = s1.strip('#')
	s2 = s2.strip('#')
	return s1 == s2


class TreeNode:
	def __init__(self, val=0, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right

def make_tree(lst):

	from queue import Queue
	q = Queue()
	i = 0
	root = TreeNode()
	root.val = lst[0]
	q.put(root)
	i = 1
	while not q.empty():
		t = q.get()
		for j in range(2):
			if i + j < len(lst):
				t = TreeNode()
				t.val = lst[i+j]
				q.put(t)
		i = i + j
		print(i)
		parent = q.get()


	# root = TreeNode()
	# root.val = lst[0]
	# q.put(root)
	# i = 1
	# k = 1
	# flag = False
	# while i < len(lst):
	# 	t = q.get()
	# 	for j in range(2**k):
	# 		if not flag:
	# 			t.left = TreeNode()
	# 		else:
	# 			t.right = TreeNode()
	# 		if i+j < len(lst):
	# 			t.val = lst[i+j]
	# 		flag = not flag
	# 		q.put(t)
	# 	k += 1
	# 	i = i + j + 1
	return root


def traverse_tree(root):
	if root:
		print(root.val)
	if root.left:
		traverse_tree(root.left)
	if root.right:
		traverse_tree(root.right)


def do_recursion(s, first, last):
	if last - first <= 1:
		return s[first] == s[last]
	elif first <= last:
		return True and do_recursion(s, first + 1, last - 1)


def validPalindrome_recu(s):
	first = 0
	last = len(s) - 1
	if len(s) == 1:
		return True
	else:
		return do_recursion(s, first, last)


def validPalindrome(s):
	first = 0
	last = -1
	l = len(s) // 2
	i = 0
	for i in range(l):
		if s[i + first] != s[last - i]:
			print(s[i + first])
			print(s[last - i])
			left = s[i + first + 1:last - i - 1:]
			right = s[i + first:last - i:]
			# validPalindrome(left)
			print(left, right)

	return True


def findMode(root):
	pass


# def levelOrderBottom(root):
#
# 	result = []
# 	if root.left:
# 		levelOrderBottom(root.left)
# 		result.append(root.left.val)
#
# 	if root.right:
# 		levelOrderBottom(root.right)
# 		result.append(root.right.val)
#
# 	total.append(result)
# 	return total

def levelOrderBottom(root):
	result = []
	if not root:
		return result
	from queue import Queue
	q = Queue()
	q.put(root)
	while not q.empty():
		t = q.get()
		pass

def floodFill(image, sr, sc, newColor):
	pass

def totalMoney(n):

	d = dict()
	d[0] = 1
	s = 0
	for i in range(1, n):
		if i % 7 == 0:
			d[i % 7] = d[i % 7] + 1
		else:
			d[i % 7] = d[(i % 7) - 1] + 1
		s += d[i % 7]
	return s + 1

def maximumUnits(boxTypes, truckSize):

	d = dict()
	for boxType in boxTypes:
		d[boxType[0]] = boxType[1]
	d = dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
	count = 0
	box = 0
	print(d)
	# l = dict()
	# for k, v in d.items():
	# 	l[v] = []
	# for k, v in d.items():
	# 	l[v].append(k)
	# for k, v in l.items():
	# 	l[k] = sorted(v, reverse=True)

	# d = dict()
	# for k, v in l.items():
	# 	for item in v:
	# 		d[item] = k
	#
	# print(d)
	# for k, v in d.items():
	# 	while k > 0 and count <= truckSize:
	# 		box += v
	# 		count += 1
	# 		k -= 1
	# return box

	# d= dict()
	# for boxType in boxTypes:
	# 	d[boxType[0]] = boxType[1]
	# d = dict(sorted(d.items(), key=lambda kv: kv[1] * kv[0], reverse=True))
	# print(d)
	# count = 0
	# box = 0
	# for k, v in d.items():
	# 	while k > 0 and count < truckSize:
	# 		box += v
	# 		count += 1
	# 		k -= 1
	# return box


def mostCommonWord(paragraph, banned):
	import re
	from collections import Counter
	paragraph = paragraph.lower()
	paragraph = re.findall(r'[~A-Za-z]+', paragraph)
	c = Counter(paragraph)
	c = sorted(c.items(), key=lambda kv: kv[1], reverse=True)
	for item in c:
		k, v = item
		if k not in banned:
			return k

def decode(encoded, first):
	result = []
	result.append(first)
	for i in range(len(encoded)):
		o = int(bin(result[i] ^ encoded[i]), 2)
		result.append(o)
	return result

def slowestKey(releaseTimes, keysPressed):

	d = dict()
	for k in keysPressed:
		d[k] = 0

	d[keysPressed[0]] = releaseTimes[0]
	for i in range(1, len(releaseTimes)):
		if releaseTimes[i] - releaseTimes[i-1] > d[keysPressed[i]]:
			d[keysPressed[i]] = releaseTimes[i] - releaseTimes[i-1]
	d = dict(sorted(d.items(), key= lambda kv: kv[1], reverse=True))
	l = dict()
	for k, v in d.items():
		l[v] = []
	for k, v in d.items():
		l[v].append(k)
	for k, v in l.items():
		l[k] = sorted(v, reverse=True)

	d = dict()
	for k, v in l.items():
		for item in v:
			d[item] = k
	k = list(d.keys())
	return k[0]

def minOperations(logs):
	stack = []
	for log in logs:
		if log == "../" and len(stack) > 0:
			stack.pop()
		elif log == "./":
			continue
		elif not log == "../":
			stack.append(log)
	print(stack)
	return len(stack)

def maxDepth(s):

	stack = []
	flag = False
	depth = 0
	for i in range(len(s)):
		if s[i] == '(':
			stack.append(s[i])
			if len(stack) > depth:
				depth = len(stack)
		elif s[i] == ')' and len(stack) > 0:
			stack.pop()
		else:
			continue
	return depth

def maxProduct(nums):

	p = 0
	for i in range(len(nums)-1):
		for j in range(i+1, len(nums)):
			if (nums[i]-1) * (nums[j]-1) > p:
				print(p)
				p = (nums[i]-1) * (nums[j]-1)
	return p

def shuffle(nums, n):

	x = []
	y = []
	for i in range(len(nums)):
		if i < n:
			x.append(nums[i])
		else:
			y.append(nums[i])
	result = []
	for i in range(len(x)):
		result.append(x[i])
		result.append(y[i])
	del x , y
	return result

def finalPrices(prices):
	result = []
	for i in range(len(prices)-1):
		flag = False
		for j in range(i+1, len(prices)):
			if prices[i] >= prices[j]:
				flag = True
				result.append(prices[i]-prices[j])
				break
		if not flag:
			result.append(prices[i])

	for i in range(len(result), len(prices)):
		result.append(prices[i])
	return result

def runningSum(nums):
	
	nums = np.array(nums)
	nums = np.cumsum(nums)
	nums = nums.tolist()
	return nums
	
def xorOperation(n, start):

	result = 0
	for i in range(n):
		result = int(bin(result ^ start+2*i), 2)
	return result

def canBeEqual(target, arr):
	return sorted(target) == sorted(arr)

def average(salary):

	salary.remove(min(salary))
	salary.remove(max(salary))
	return np.mean(salary)

# def isPathCrossing(path):
	# d = dict({'N': (0, 1), 'S': (0, -1), 'E': (-1, 0), 'W': (1, 0)})
	# t = (0, 0)
	# i0, j0 = t
	# for letter in path:
	# 	i1, j1 = d[letter]
	# 	i0 += i1
	# 	j0 += j1
	# 	print(i0, j0)
	# 	if i0 == 0 and j0 == 0:
	# 		return True
	# return False

def isPathCrossing(path):

	direction = { 'N' : (0, 1),
                  'S' : (0, -1),
                  'E' : (-1, 0),
                  'W' : (1, 0) }
	x, y = 0, 0
	trajectory = {(0, 0)}
	for i in path:
		x = direction[i][0] + x
		y = direction[i][1] + y
		if (x,y) in trajectory:
			return True
		else:
			trajectory.add((x,y))
		print(trajectory)
		return False


def numberOfMatches(n):
	result = 0
	if n == 2:
		return 1
	elif n % 2 == 0:
		result = result + (n // 2) + numberOfMatches(n // 2)
	elif n % 2 == 1:
		result = result + (n // 2) + numberOfMatches((n - 1) // 2 + 1)
	return result


def interpret(command):
	d = dict({'G': 'G', '()': 'o', '(al)': 'al'})
	for k, v in d.items():
		command = command.replace(k, v)
	return command


def maximumWealth(accounts):
	l = list()
	for account in accounts:
		l.append(sum(account))
	return max(l)


def maxRepeating(sequence, word):
	start = 0
	count = 0
	end = -1
	if len(sequence) == 1:
		end = 1
	while sequence.find(word, start, end) > -1:
		count += 1
		start = start + len(word)
	return count


def arrayStringsAreEqual(word1, word2):
	s1 = "".join(word1)
	s2 = "".join(word2)
	return s1 == s2


def numIdenticalPairs(nums):
	import numpy as np
	args = np.argsort(nums)
	count = 0
	i = 0
	while i < (len(args) - 1):
		for j in range(i + 1, len(args)):
			if nums[args[i]] == nums[args[j]]:
				count += 1
			else:
				break
		i += 1
	return count


# def countOdds(low, high):
# 	count = len(list(filter(lambda x: x%2==1, range(low, high+1))))
# 	return count

def countOdds(low, high):
	diff = high - low + 1
	if diff % 2 == 0:
		return diff // 2

	if high % 2 == 1:
		return 1 + diff // 2
	return diff // 2


def restoreString(s, indices):
	s = list(s)
	result = [''] * len(s)
	for i in range(len(s)):
		result[indices[i]] = s[i]
	return "".join(result)


# end = arr[-1]
# result = list(filter(lambda x: x not in arr,range(end)))
# return result[k]

def findKthPositive(arr, k):
	l = k
	# start
	index = []
	if arr[0] != 1:
		arr.insert(0, 1)
		index.append(1)
		k -= 1

	# mid
	for i in range(len(arr) - 1):
		if arr[i + 1] - arr[i] > 1:
			[index.append(j) for j in range(arr[i] + 1, arr[i + 1])]
			k -= (arr[i + 1] - arr[i] - 1)
	print(len(index), index)
	if len(index) >= l:
		return index[l - 1]
	if len(index) == 0:
		return arr[-1] + l
	l -= len(index)
	return arr[-1] + l


def modifyString(s):
	import random
	s = list(s)
	a = set(chr(ord('a') + i) for i in range(0, 26))
	if len(s) == 1 and s[0] == '?':
		s[0] = 'a'
		return "".join(s)

	for i in range(len(s)):
		r = set()
		if s[i] == '?':
			if i == 0:
				r.add(s[i + 1])
			if 0 < i < len(s) - 1:
				f = s[i - 1]
				e = s[i + 1]
				r.add(f)
				r.add(e)
			if i == len(s) - 1:
				r.add(s[i - 1])

			r = a.difference(r)
			c = random.choice(list(r))
			s[i] = c
	return "".join(s)


def makeGood(s):
	s = list(s)
	r = str()
	flag = False
	for i in range(len(s) - 1):
		if s[i] == str.lower(s[i + 1]):
			s[i] = ''
			s[i + 1] = ''
			makeGood("".join(s))
			break
	if not flag:
		flag = True
		print(s)
	return r


def countGoodTriplets(arr, a, b, c):
	count = 0
	for i in range(len(arr) - 2):
		for j in range(i + 1, len(arr) - 1):
			for k in range(j + 1, len(arr)):
				if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
					count += 1
	return count


def countConsistentStrings(allowed, words):
	count = 0
	allowed = set(allowed)
	for word in words:
		word = set(word)
		if word.issubset(allowed):
			count += 1
	return count


# def countConsistentStrings(allowed, words):
# 	count = 0
# 	for word in words:
# 		flag = True
# 		for letter in word:
# 			if letter not in word:
# 				flag = False
# 				break
# 		if flag:
# 			count += 1
# 	return count

def containsPattern(arr, m, k):
	pass


def trimMean(arr):
	import numpy as np
	arr = sorted(arr)
	percent = [5, 95]
	for i in range(len(percent)):
		percent[i] = int(np.floor(percent[i] / 100 * len(arr)))
	arr = arr[percent[0] + 1:percent[-1] - 1]
	return np.mean(arr)


def choose_variants(i, valid, invalid):
	if i in valid:
		invalid.add(i)
	else:
		valid.add(i)
	return valid, invalid


def numSpecial(mat):
	import numpy as np
	row, column = np.shape(mat)
	valid_r = set()
	invalid_r = set()
	valid_c = set()
	invalid_c = set()
	for i in range(row):
		for j in range(column):
			if mat[i][j] == 1:
				valid_r, invalid_r = choose_variants(i, valid_r, invalid_r)
				valid_c, invalid_c = choose_variants(j, valid_c, invalid_c)

	valid_r = valid_r.difference(invalid_r)
	valid_c = valid_c.difference(invalid_c)
	count = 0
	for i in valid_r:
		for j in valid_c:
			if mat[i][j] == 1:
				count += 1
	return count


def reorderSpaces(text):
	import re
	s = re.findall('\\s', text)
	w = re.findall('[^\\s]+', text)

	r = 0
	num_space = 0
	if len(w) > 1:
		num_space = len(s) // (len(w) - 1)
		r = len(s) % (len(w) - 1)
	else:
		r = len(s)

	result = []
	for j in range(len(w)):
		result.append(w[j])
		for i in range(num_space):
			result.append(' ')

	if len(w) > 1:
		result = result[:-num_space]
	for i in range(r):
		result.append(' ')
	return "".join(result)


def maxLengthBetweenEqualCharacters(s):
	result = -1
	for i in range(len(s) - 1):
		for j in range(1, len(s)):
			if s[i] == s[j]:
				if j - i - 1 > result:
					result = j - i - 1
	return result


def frequencySort(nums):
	from collections import Counter
	c = Counter(nums)
	c = dict(sorted(c.items(), key=lambda kv: kv[1]))
	v = np.array(list((c.values())))
	k = list(c.keys())
	print(k, v)

	u = sorted(set(v))
	result = []
	for i in u:
		tmp = np.where(v == i)[0]
		l = []
		for j in tmp:
			l.append(k[j])
		l = sorted(l, reverse=True)
		[result.append(k) for k in l]

	print(result)
	output = np.repeat(result, v)
	return list(output)


def getMaximumGenerated(n):
	if n == 0:
		return 1
	result = []
	result.append(0)
	result.append(1)
	for i in range(2, n + 1):
		if i % 2 == 0:
			result.append(result[i // 2])
		else:
			result.append(result[i // 2] + result[i // 2 + 1])
	result = sorted(result, reverse=True)
	return result[0]


def rotatedDigits(N):
	s1 = {'0', '1', '8'}
	s2 = {'6', '9', '2', '5'}
	count = 0
	for num in range(1, N + 1):
		num = set(str(num))
		if num.issubset(s2) or (num.intersection(s2) != set() and num.difference(s2).issubset(s1)):
			count += 1
	return count


def canFormArray(arr, pieces):
	for j in arr:
		i = -1
		tmp = []
		for piece in pieces:
			i += 1
			if j in piece:
				if arr[i:len(piece) + i] != piece:
					return False
				pieces.remove(piece)
				for _ in range(piece):
					arr.remove(arr[i])
					i += 1
				continue
	return True


def canRecursive(arr, pieces, i):

	if i < len(pieces):
		if pieces[i][0] in arr:
			start = np.where(arr == pieces[i])[0][0]
			l = [start + j for j in range(len(pieces[i]))]
			if l[-1] > len(arr) - 1:
				return False
			if arr.take(l) != pieces[i]:
				return False
			return True and canRecursive(arr, pieces, i + 1)

	return True

def canFormArray(arr, pieces):

	# i = 0
	# l = np.array(arr)
	# return canRecursive(l, pieces, i)
	arr = np.array(arr)
	for piece in pieces:
		if piece[0] not in arr:
			return False
		index = np.where(arr == piece[0])[0][0]
		l = arr[index: index + len(piece)].tolist()
		if l != piece:
			return False
	return True

# def minTimeToVisitAllPoints(points):
#
# 	m = 10000
# 	for i in range(len(points)):
# 		count = 0
# 		for j in range(i, i+len(points)-1):
# 			f, s = j, j+1
# 			f = f % len(points)
# 			s = s % len(points)
# 			x1, y1 = points[f]
# 			x2, y2 = points[s]
# 			if x2-x1 == y2-y1:
# 				count += abs(x2-x1)
# 			else:
# 				count += abs(x2-x1)
# 				count += abs(y2-y1)
# 				print(count, m)
# 			if count < m:
# 				m = count
# 	return m

def minTimeToVisitAllPoints(points):
	pass
	# count = 0
	# for i in range(len(points)-1):
	# 	x1, y1 = points[i+1]
	# 	x0, y0 = points[i]
	# 	while (x1-x0):
	#
	#
	# 	if x2-x1 == y2-y1:
	# 		count += abs(x2-x1)
	# 	else:
	# 		count += abs(x2-x1)
	# 		count += abs(y2-y1)
	# 		print(count)
	# return count


def isMonotonic(A):

	flag_I = False
	flag_D = False
	for i in range(len(A)-1):
		if flag_D and A[i+1] > A[i]:
			return False
		if flag_I and A[i+1] < A[i]:
			return False

		if A[i+1] > A[i]:
			flag_I = True
			flag_D = False
		if A[i+1] < A[i]:
			flag_I = False
			flag_D = True

	return True

def process(index, result, source, dest, total):

	if result[index] == -1000:
		for i in source:
			if total - i in dest:
				result[index] = i
				break
	return result

def fairCandySwap(A, B):

	mean = (sum(A) + sum(B)) // 2
	A = sorted(A, reverse=True)
	B = sorted(B, reverse=True)

	sumA = sum(A)
	sumB = sum(B)

	result = []
	for i in A:
		diff = abs(mean-(sumA-i))
		for j in B:
			if j == diff:
				result.append(i)
				result.append(j)
				return result
	return result

def prepare(l):
	c = Counter(l)
	c = dict(sorted(c.items(), key=lambda kv: kv[0]))
	return c

def shortestCompletingWord(licensePlate, words):

	licensePlate = re.findall('[A-Za-z]+', licensePlate)
	licensePlate = "".join(licensePlate).lower()
	pattern = prepare(licensePlate)
	pattern_keys = set(pattern.keys())

	d = dict()
	for i, word in enumerate(words):
		d[i] = prepare(word)

	result = []
	for key, value in d.items():
		keys = set(value.keys())
		if not pattern_keys.issubset(keys):
			result.append(key)

	for k in result:
		d.pop(k)

	for key, value in d.items():
		count = 0
		for k, v in value.items():
			if not pattern.keys().__contains__(k):
				continue
			if v >= pattern[k]:
				count += v
		d[key] = count

	l = sum(pattern.values())
	d = dict(filter(lambda kv: kv[1] >= l, d.items()))
	d = dict(sorted(d.items(), key=lambda kv: kv[1]))

	keys = list(d.keys())
	values = list(d.values())
	if len(values) < 1:
		return words[keys[0]]

	result = []
	for k, v in d.items():
		result.append(words[k])
	result = sorted(result, key=lambda x: len(x))
	return result[0]

def numEquivDominoPairs(dominoes):

	l = list()
	for dominoe in dominoes:
		dominoe = tuple(sorted(dominoe))
		l.append(dominoe)

	c = Counter(l)
	count = 0
	for k, v in c.items():
		while v > 1:
			count += v - 1
			v = v - 1
	return count

# Definition for a binary tree node.
class TreeNode:
	def __init__(self, val=0, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right

	def buildTree(self, l):
		pass

	def leafSimilar(self, root1, root2):
		result = []
		if root1 is not None:
			self.leafSimilar(root1.left)
			result.append(root1.val)
			self.leafSimilar(root1.right)

def isPalindrome(t):
	start = 0
	end = -1
	for i in range((len(t)+1) //2):
		if t[start] != t[end]:
			return False
		start += 1
		end -= 1
	return True


def recursive(s, start, end, error):

	result = True
	if start < end:
		if s[start] == s[end]:
			return result and recursive(s, start+1, end-1, error)
		elif error < 1:
			result = True
		elif error > 1:
			result = False

		return result and recursive(s, start+1, end, error+1)
		return result and recursive(s, start, end-1, error+1)
	return result

def validPalindrome(s):

	# start = 0
	# end = -1
	# for i in range(len(s) // 2):
	# 	if s[start] != s[end]:
	# 		return False
	# 	start += 1
	# 	end -= 1
	# return True
	start = 0
	end = len(s) - 1
	error = 0
	return recursive(s, start, end, error)

def repeatedSubstringPattern(s):

	pattern = []
	j = 1
	pattern.append(s[0])
	while j < len(s) and s[0] != s[j]:
		pattern.append(s[j])
		if len(pattern) > len(s) // 2:
			return False
		j += 1

	i = 0
	for k in range(j, len(s)):
		if s[k] != pattern[i]:
			return False
		i = (i + 1) % len(pattern)
	return True

def numUniqueEmails(emails):

	result = set()
	for email in emails:
		name = re.split('@.+', email)[0]
		domain = re.findall('@.+', email)[0]
		name = re.sub('[\.]', '', name)
		name = re.sub('\+\w+', '', name)
		name = name + domain
		result.add(name)
	return len(result)

def isRectangleOverlap(rec1, rec2):
	pass

def isLongPressedName(name, typed):

	i = 0
	n = 0
	sub = ""
	name = list(name)
	typed = list(typed)
	while i < len(name)-1:
		sub = name[i]
		j = 1
		while i + j < len(name) and name[i] == name[i+j]:
			sub += name[i]
			j += 1
		k = 1
		com = typed[n]
		while n + k < len(typed) and typed[n] == typed[n+k]:
			com += typed[k]
			k += 1
		print(sub, com)
		i = i + j
		n = n + k


def canMakeArithmeticProgression(arr):

	diff = set()
	arr = sorted(arr)
	for i in range(len(arr)-1):
		diff.add(arr[i+1]-arr[i])
	print(diff)
	return len(diff) == 1

def reformatDate(date):
	day = dict({'1st': '01', '2nd': '02', '3rd': '03', '4th': '04', '5th': '05', '6th':'06', '7th':'07','8th':'08', '9th':'09', '10th':'10', '11th':'11', '12th':'12', '13th':'13', '14th':'14', '15th':'15', '16th':'16', '17th':'17', '18th':'18', '19th':'19', '20th':'20', '21th':'21', '22th': '22', '23th': '23','24th': '24','25th': '25','26th': '26','27th': '27','28th': '28', '29th': '29','30th': '30', '31th': '31'})
	month = dict({"Jan": '01', "Feb": '02', "Mar": '03', "Apr": '04', "May": '05', "Jun":'06', "Jul":'07', "Aug":'08', "Sep":'09', "Oct":'10', "Nov":'11', "Dec":'12'})

	d = date.split()
	result = []
	result = d[-1]+'-'+month[d[1]]+'-'+day[d[0]]
	return result

def removePalindromeSub(s):

	for letter in s:
		pass

def tribonacci(n):
	result = []
	result.append(0)
	result.append(1)
	result.append(1)
	for i in range(3, n+1):
		result.append(result[i-3] + result[i-2] + result[i-1])
	return result[-1]

def distanceBetweenBusStops(distance, start, destination):

	f_d = 1000
	if start < destination:
		f_d = sum(distance[start:destination])

	tmp = start
	start = destination
	destination = tmp

	print(start, destination)
	b_d = distance[start]
	for i in range(1, len(distance)):
		d = (start + i) % len(distance)
		if d == destination:
			break
		b_d += distance[d]
	print(f_d, b_d)
	return min(f_d, b_d)



if __name__ == "__main__":
	# S = "z"
	# shifts = [52]
	# S = shiftingLetters(S, shifts)
	# print(S)
	# nums = [0,1,2,4,5,7]
	# result = summaryRanges(nums)
	# print(result)

	# obj = NumArray([-2, 0, 3, -5, 2, -1])
	# val0 = obj.sumRange(0, 2)
	# val1 = obj.sumRange(2, 5)
	# val2 = obj.sumRange(0, 5)
	# print(val0, val1, val2)

	# result = trailingZeroes(10)
	# print(result)

	# S = "y#fo##f"
	# T = "y#f#o##f"
	# result = backspaceCompare(S, T)
	# print(result)

	# n = trailingZeroes(50)
	# print(f'Total is {n}')
	#
	# root = TreeNode()
	# root.val = 3
	# root.left.val = 9
	# root.right.val = 20
	# left_root = root.left
	# left_root.left = None
	# left_root.right = None
	# right_root = root.right
	# right_root.left.val = 15
	# right_root.right.val = 7
	# right_root.left.left = None
	# right_root.right.right = None
	# lst = [3, 9, 20, None, None, 15, 7]
	# root = make_tree(lst, 0)
	# traverse_tree(root)
	# result = levelOrderBottom(root)
	# lst = [1, None, 2, 2]
	# root = make_tree(lst, 0)
	# result = findMode(root)
	# stri = 'abca'
	# result = validPalindrome(stri)
	# print(result)

	# paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
	# banned = ['hit']
	# result = mostCommonWord(paragraph, banned)
	# print(result)
	#
	# target = [1,2,3,4]
	# arr = [2,4,1,3]
	# result = canBeEqual(target, arr)
	# print(result)
	#
	# result = numberOfMatches(14)
	# print(result)

	# command = "(al)G(al)()()G"
	# result = interpret(command)
	# print(result)

	# accounts = [[1,2,3],[3,2,1]]
	# result = maximumWealth(accounts)
	# print(result)

	# word1 = ["ab", "c"]
	# word2 = ["a", "bc"]
	# result = arrayStringsAreEqual(word1, word2)
	# print(result)

	# nums = [1,2,3,1,1,3]
	# result = numIdenticalPairs(nums)
	# print(result)

	# count = countOdds(3, 7)
	# print(count)

	# s = "codeleet"
	# indices = [4,5,6,7,0,2,1,3]
	# result = restoreString(s, indices)
	# print(result)

	# arr = [1,1,2,2,3]
	# a = 0
	# b = 0
	# c = 1
	# result = countGoodTriplets(arr, a, b, c)
	# print(result)
	# arr = [2, 3, 4, 7, 11]
	# arr = [1,2,3,4]
	# k = 5
	# arr = [1,13,18]
	# k =17
	# result = findKthPositive(arr, k)
	# print(result)

	# s = "?zs"
	# s = modifyString(s)
	# print(s)

	# s = "abBAcC"
	# s = makeGood(s)
	# print(s)

	# arr = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2]
	# result = trimMean(arr)
	# print(result)

	# mat = [[1,0,0], [0,1,0], [0,0,1]]
	# result = numSpecial(mat)
	# print(result)

	# text = "  this   is  a sentence "
	# text = " practice   makes   perfect"
	# text = "a"
	# text = "  hello"
	# r = reorderSpaces(text)
	# print(r)

	# s = "aa"
	# result = maxLengthBetweenEqualCharacters(s)
	# print(result)

	# nums = [1,2,2,3,3]
	# nums = frequencySort(nums)

	# n = 3
	# result = getMaximumGenerated(n)
	# print(result)

	# code = [5,7,1,4]
	# k = 3
	# result = decrypt(code, k)
	# print(result)

	# N = 857
	# result = rotatedDigits(N)
	# print(result)

	# arr = [15,88]
	# pieces = [[88],[15]]
	# result = canFormArray(arr, pieces)
	# print(result)

	# nums = [-1,1,-6,4,5,-6,1,4,1]
	# nums = [-39, 27, 27, -11, -39, -11, -11, 27, -11, -26, -33, -26, -11, 0, -11, 0, -26, 27, -39, -26, 0, 27, -33, -33,
	#         27, 0, 27, 27, -33, 0, -11, -26, -11]
	# result = frequencySort(nums)
	# print(result)

	# arr = [49,18,16]
	# pieces = [[16,18,49]]
	# arr = [15,88]
	# pieces = [[88],[15]]
	# arr = [91,4,64,78]
	# pieces = [[78],[4,64],[91]]
	# arr = [1,2,3]
	# pieces = [[2],[1,3]]
	# result = canFormArray(arr, pieces)
	# print(result)

	# points = [[1,1],[3,4],[-1,0]]
	# result = minTimeToVisitAllPoints(points)
	# print(result)

	# arr = [1,2,2,3]
	# arr = [1,3,2]
	# arr = [11,11,9,4,3,3,3,1,-1,-1,3,3,3,5,5,5]
	# arr = [1,1,1]
	# arr = [1,1,0]
	# result = isMonotonic(arr)
	# print(result)

	# A = [1,1]
	#  B = [2,2]
	# A = [2]
	# B = [1,3]
	# A = [1,2,5]
	# B = [2,4]
	# result = fairCandySwap(A, B)
	# print(result)


	# licensePlate = "1s3 PSt"
	# words = ["step","steps","stripe","stepple"]
	# licensePlate = "1s3 456"
	# words = ["looks","pest","stew","show"]
	# licensePlate = "AN87005"
	# words = ["participant","individual","start","exist","above","already","easy","attack","player","important"]
	# licensePlate = "e490936"
	# words = ["involve","those","else","violence","six","positive","product","expect","close","couple"]
	# licensePlate = "TE73696"
	# words = ["ten","two","better","talk","suddenly","stand","protect","collection","about","southern"]
	# licensePlate = "T713264"
	# words = ["executive","suffer","product","language","manager","thought","various","attention","how","medical"]
	# result = shortestCompletingWord(licensePlate, words)
	# print(result)

	# points = [[1,1],[3,4],[-1,0]]
	# result = minTimeToVisitAllPoints(points)
	# print(result)

	# A = [1,1]
	# B = [2,2]
	# A = [1,2]
	# B = [2,3]
	# A = [1,2,5]
	# B = [2,4]
	# result = fairCandySwap(A, B)
	# print(result)

	# dominoes = [[1,2],[2,1],[3,4],[5,6]]
	# dominoes = [[1,2],[1,2],[1,1],[1,2],[2,2]]
	# result = numEquivDominoPairs(dominoes)
	# print(result)
	# t = TreeNode()

	# s = "abca"
	# # s = "cbbcc"
	# result = validPalindrome(s)
	# print(result)

	# s = "ababab"
	# result = repeatedSubstringPattern(s)
	# print(result)

	# lst = [3,9,20,0,0,15,7]
	# root = make_tree(lst)
	# traverse_tree(root)


	root = TreeNode(3)
	root.left = TreeNode(9)
	root.right = TreeNode(20)
	left = root.left
	right = root.right
	right.left = TreeNode(15)
	right.right = TreeNode(7)
	# traverse_tree(root)
	# global total
	# total = []
	# result = levelOrderBottom(root)
	# print(result)

	# result = totalMoney(20)
	# print(result)
	#
	# boxTypes = [[1,3],[2,2],[3,1]]
	# truckSize = 4

	# boxTypes = [[5,10],[2,5],[4,7],[3,9]]
	# truckSize = 10
	# boxTypes = [[2,1],[4,4],[3,1],[4,1],[2,4],[3,4],[1,3],[4,3],[5,3],[5,3]]
	# truckSize = 13
	# result = maximumUnits(boxTypes,truckSize)
	# print(result)
	# encoded = [1,2,3]
	# first = 1
	# encoded = [6,2,7,3]
	# first = 4
	# result= decode(encoded, first)
	# print(result)

	# releaseTimes = [9,29,49,50]
	# keysPressed = "cbcd"
	# result = slowestKey(releaseTimes, keysPressed)
	# print(result)

	# logs = ["d1/","d2/","./","d3/","../","d31/"]
	# print(logs)
	# result = minOperations(logs)
	# print(result)

	# s = "(1+(2*3)+((8)/4))+1"
	# s = "(1)+((2))+(((3)))"
	# s = "1+(2*3)/(2-1)"
	# result = maxDepth(s)
	# print(result)

	# nums = [3,4,5,2]
	# result = maxProduct(nums)
	# print(result)

	# nums = [1,2,3,4,4,3,2,1]
	# n = 4
	# result = shuffle(nums, n)
	# print(result)
	# prices = [8,4,6,2,3]
	# prices = [4,7,1,9,4,8,8,9,4]
	# result = finalPrices(prices)
	# print(result)
	# nums = [1,2,3,4]
	# result = runningSum(nums)
	# print(result)

	# n = 5
	# start = 0
	# result = xorOperation(n, start)
	# print(result)

	# salary = [1000,2000,3000]
	# result = average(salary)
	# print(result)

	# path = "NESWW"
	# path = "NNSWWEWSSESSWENNW"
	# path = 'NES'
	# result = isPathCrossing(path)
	# print(result)

	# root1 = make_tree([3,5,1,6,2,9,8,None,None,7,4])
	# traverse_tree(root1)

	# emails = ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
	# result = numUniqueEmails(emails)
	# print(result)

	# name = "alex"
	# name = "saeeddd"
	# name = "saeed"
	# # typed = "aaleex"
	# typed = "ssaeed"
	# result = isLongPressedName(name, typed)
	# print(result)

	# arr = [3,5,1]
	# result = canMakeArithmeticProgression(arr)
	# print(result)

	# date = "20th Oct 2052"
	# result = reformatDate(date)
	# print(result)

	# s = "ababa"
	# result = removePalindromeSub(s)
	# print(result)

	# n = 4
	# result = tribonacci(n)
	# print(result)

	# distance = [1,2,3,4]
	# start = 0
	# destination = 3
	# distance = [1,2,3,4]
	# start = 0
	# destination = 2
	#
	# distance = [8,11,6,7,10,11,2]
	# start = 0
	# destination = 3


	distance = [7,10,1,12,11,14,5,0]
	start = 7
	destination = 2
	result = distanceBetweenBusStops(distance, start, destination)
	print(result)