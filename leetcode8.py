from itertools import combinations
from random import random

import numpy as np
import pandas as pd
import re
import builtins
from collections import Counter
import copy

def letterCombinations(digits: str):

	import copy
	d = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'], '5': ['j', 'k', 'l'], '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'], '8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}
	def dfs(digits, result, position=0, tmp=[], counter=0):
		if len(digits) == 0:
			return ""

		if position < len(digits):
			for c in d[digits[position]]:
				if len(digits) - counter > 0:
					tmp.append(c)
					dfs(digits, result, position+1, tmp, counter+1)
					if len(tmp) == len(digits):
						result.append(copy.deepcopy("".join(tmp)))
					tmp.pop()
					counter -= 1
	result = []
	dfs(digits, result)
	return result

# digits = "258"
# result = letterCombinations(digits)
# print(result)

def findLengthOfLCIS(nums):
	if len(nums) < 1:
		return nums

	index = 0
	d = {index: 1}
	for i in range(1, len(nums)):
		if nums[i] > nums[i - 1]:
			d[index] += 1
		else:
			index = i
			d[index] = 1
	d = dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
	k = list(d.keys())
	v = list(d.values())
	return nums[k[0]:v[0]]

# nums = [1,3,5,4,7]
# result = findLengthOfLCIS(nums)
# print(result)

def kthSmallest(matrix, k: int) -> int:
	import numpy as np
	array = np.ravel(matrix)
	array = sorted(array)
	return array[k-1]

# matrix = [
#    [ 1,  5,  9],
#    [10, 11, 13],
#    [12, 13, 15]
# ],
# k = 8
# result = kthSmallest(matrix, k)
# print(result)

def findDuplicates(nums):
	# result = set(filter(lambda x: nums.count(x) == 2, nums))
	# return list(result)
	from collections import Counter
	c = Counter(nums)
	result = []
	for k, v in c.items():
		if v == 2:
			result.append(k)
	return result
# nums = [4,3,2,7,8,2,3,1]
# result = findDuplicates(nums)
# print(result)

def processQueries(queries, m: int):

	p = list(range(1, m+1))
	result = []
	for query in queries:
		for i in range(len(p)):
			if query == p[i]:
				result.append(i)
				p.remove(query)
				p.insert(0, query)
				break
	return result


# queries = [3,1,2,1]
# m = 5
# queries = [4,1,2,2]
# m = 4
# result = processQueries(queries, m)
# print(result)

def findAndReplacePattern(words, pattern):
	def match(word):
		m1, m2 = {}, {}
		for w, p in zip(word, pattern):
			print(w, p)
			if w not in m1: m1[w] = p
			if p not in m2: m2[p] = w
			if (m1[w], m2[p]) != (p, w):
				return False
		return True
	return list(filter(match, words))


# words = ["abc","deq","mee","aqq","dkd","ccc"]
# pattern = "abb"
# result = findAndReplacePattern(words, pattern)
# print(result)

def groupThePeople(groupSizes):

	d = dict()
	for i in range(len(groupSizes)):
		if groupSizes[i] not in d:
			d[groupSizes[i]] = [i]
		else:
			d[groupSizes[i]].append(i)

	result = []
	for k, v in d.items():
		if len(v) <= k:
			result.append(v)
		else:
			for i in range(0, len(v),k):
				result.append(v[i:i+k])
	return result


# groupSizes = [3,3,3,3,3,1,3]
# groupSizes = [2,1,3,3,3,2]
# result = groupThePeople(groupSizes)
# print(result)

def maxCoins(piles):

	l = len(piles) // 3
	piles = sorted(piles, reverse=True)
	count = 0
	i = 1
	for _ in range(l):
		count += piles[i]
		i += 2
	return count

# piles = [2,4,1,2,7,8]
# piles = [9,8,7,6,5,1,2,3,4]
# result = maxCoins(piles)
# print(result)

def maxWidthOfVerticalArea(points):
	points = sorted(points, key=lambda kv: kv[0])
	print(points)
	m = 0
	for i in range(1, len(points)):
		m = max(points[i][0]-points[i-1][0], m)
	return m

# points = [[8,7],[9,9],[7,4],[9,7]]
# points =  [[3,1],[9,0],[1,0],[1,4],[5,3],[8,8]]
# result = maxWidthOfVerticalArea(points)
# print(result)

def minAddToMakeValid(S: str):
	from collections import deque
	st = []
	S = list(S)
	for i in range(len(S)):
		if S[i] == '(':
			st.append(S[i])
		elif len(st) > 0 and st[-1] == '(' and S[i] == ')':
			st.pop()
		else:
			st.append(S[i])

	return len(st)

# S = "())"
# S = "((("
# result = minAddToMakeValid(S)
# print(result)

def isSeries(nums):
	for i in range(1, len(nums)-1):
		if nums[i] - nums[i-1] != nums[i+1] - nums[i]:
			return False
	return True

def checkArithmeticSubarrays(nums, l, r):
	result = []
	for i in range(len(l)):
		arr = sorted(nums[l[i]:r[i]+1], reverse=True)
		result.append(isSeries(arr))
	return result

# nums = [4,6,5,9,3,7]
# l = [0,0,2]
# r = [2,3,5]
# result = checkArithmeticSubarrays(nums, l, r)
# print(result)

def bitwiseOr(arr):
	if len(arr) <= 1:
		return arr[0]
	result = 0
	for i in range(len(arr)):
		result |= arr[i]
	return result

def subarrayBitwiseORs(arr):
	def subArrays(arr, start, end):
		#mid point
		if start == end:
			return
		mid = (start + end) // 2
		subArrays(arr, start, mid)
		subArrays(arr, mid+1, end)
		print(arr[start:end+1])
	subArrays(arr, 0, len(arr)-1)


# arr = [1,3, 4, 5, 7]
# result = subarrayBitwiseORs(arr)
# print(result)

def perm(arr):
	from itertools import combinations
	result = set()
	for i in range(1, len(arr)+1):
		p = combinations(arr,i)
		for j in p:
			print(j)
			result.add(bitwiseOr(j))
	return len(result)

# arr = [1, 2, 4]
# result = perm(arr)
# print(result)

def subarrayBitwiseORs(arr) -> int:

	result = set()
	def subArrays(arr, start, end):

		# Stop if we have reached the end of the array
		if end == len(arr):
			return

		# Increment the end point and start from 0
		elif start > end:
			return subArrays(arr, 0, end + 1)

		# Print the subarray and increment the starting
		# point
		else:
			result.add(tuple(arr[start:end+1]))
			return subArrays(arr, start + 1, end)
	subArrays(arr, 0, 0)
	return result

# arr = [1,1,1]
# k = 2
# result = subarrayBitwiseORs(arr)
# print(result)

def recursive(arr):
	def dfs(arr, index):
		if index > len(arr):
			return
		for i in range(index+1, len(arr)+1):
			print(arr[index:i])
		dfs(arr, index+1)
	dfs(arr, 0)

# arr = [1, 2, 4]
# arr = [1,2,1,2,1]
# result = recursive(arr)
# print(result)


def maxProduct(nums) -> int:

	global_max = float('-inf')
	for i in range(len(nums)):
		global_max = max(nums[i], global_max)
		local_max = nums[i]
		for j in range(i+1, len(nums)):
			local_max = local_max * nums[j]
			global_max = max(local_max, global_max)
	return global_max

# nums = [2,3,-2,4, 0, 0]
# nums = [-2,0,-1]
# nums = [-2,3,-4]
# result = maxProduct(nums)
# print(result)

def Kadenas(nums):
	if nums is []:
		return 0
	global_max = local_min = local_max = nums[0]
	for i in nums[1:]:
		tmp = local_max
		local_max = max(i, i*local_max, i*local_min)
		local_min = min(i, i*local_min, i*tmp)
		global_max = max(global_max, local_max)
	return global_max

# # nums = [2,3,-2,4, 0, 0]
# # nums = [-2,3,-4]
# nums = [-2, -3, 4]
# # nums = [-6, 4, 5, 6, -2 ]
# nums = [-2,3,-4]
# result = Kadenas(nums)
# print(result)


def maxTurbulenceSize(arr) -> int:
	global_count = 0
	local_count = 1
	i = 1
	mini = float('inf')
	maxi = float('-inf')
	while i < len(arr):
		if i % 2 == 1:
			maxi = max(arr[i], arr[i-1])
		else:
			mini = min(arr[i], arr[i-1])
		print(maxi, mini)
		if mini == maxi:
			local_count += 1
			global_count = max(global_count, local_count)
		else:
			local_count = 1
		i += 1
	return global_count
# arr = [9,4,2,10,7,8,8,1,9]
# # arr = [4,8,12,16]
# # arr = [100]
# result = maxTurbulenceSize(arr)
# print(result)

def oldfindMaxAverage(nums, k) -> float:
	global_max = float('-inf')
	for i in range(len(nums)-k+1):
		global_max = max(global_max, sum(nums[i:i+k]))
	return global_max / k

def findMaxAverage(nums, k) -> float:
	global_max = sum(nums[:k])
	local_max  = global_max
	j = 0
	for i in range(k, len(nums)):
		local_max = local_max - nums[j] + nums[i]
		global_max = max(global_max, local_max)
		print(global_max)
		j += 1
	return global_max / k

# nums = [1,12,-5,-6,50,3]
# # nums = [0,4,0,3,2]
# k = 4
# # k = 1
# result = findMaxAverage(nums, k)
# print(result)



# nums = [1,2,3]
# k = 3
# nums = [1,1,1]
# k = 2
# nums = [1]
# k = 1
# nums = [1,-1,0]
# k = 0
# nums = [1,2,1,2,1]
# k = 3
# result = subarraySum(nums, k)
# print(result)

def generateSubsetsRecursively(nums, k) -> int:
	def dfs(arr, index):

		if index > len(arr):
			return 0
		for i in range(index+1, len(arr)+1):
			if sum(arr[index:i]) == k:
				print(arr[index:i])
		dfs(arr, index + 1)
	count = 0
	dfs(nums, 0)
	print(count)

# nums = [1,2,1,2,1]
# k = 7
# result = generateSubsetsRecursively(nums, k)
# print(result)

def xorQueries(arr, queries):
	result = []
	for query in queries:
		val = 0
		for i in arr[query[0]:query[1]+1]:
			val ^= i
		result.append(val)
	return result

# arr = [1,3,4,8]
# queries = [[0,1],[1,2],[0,3],[3,3]]
# arr = [4,8,2,10]
# queries = [[2,3],[1,3],[0,0],[0,3]]
# result = xorQueries(arr, queries)
# print(result)

def maxSubArray(nums):
	max_so_far = nums[0]
	curr_max = nums[0]
	d = dict()
	prev_max = float('-inf')
	start = 0
	last = 0
	for i in range(1, len(nums)):
		curr_max = max(nums[i], curr_max + nums[i])
		max_so_far = max(max_so_far, curr_max)
	return max_so_far

# nums = [-2,1,-3,4,-1,2,1,-5,4]
# result = maxSubArray(nums)
# print(result)

def subarraySum(nums, k) -> int:

	result = []
	def subArrays(arr, start, end):

		# Stop if we have reached the end of the array
		if end == len(arr):
			return

		# Increment the end point and start from 0
		elif start > end:
			subArrays(arr, 0, end + 1)

		# Print the subarray and increment the starting
		# point
		else:
			result.append(arr[start:end+1])
			subArrays(arr, start + 1, end)
	subArrays(nums, 0, 0)
	count = 0
	for item in result:
		if sum(item[:]) == k:
			count += 1
	return count

# nums = [1,1,1]
# k = 2
# result = subarraySum(nums, k)
# print(result)

def numTilePossibilities(tiles: str) -> int:
	from itertools import permutations
	tiles = list(tiles)
	result = set()
	for i in range(1, len(tiles)+1):
		p = permutations(tiles, i)
		for j in p:
			result.add("".join(j))
	return len(result)
#
# tiles = "AAB"
# result = numTilePossibilities(tiles)
# print(result)

def totalHammingDistance(nums) -> int:
	count = 0
	for i in range(32):
		count_zero = 0
		count_one = 0
		for n in nums:
			temp = n & (1 << i)
			if not temp:
				count_zero += 1
			else:
				count_one += 1
		count += (count_zero * count_one)
	return count

# nums = [4, 14, 2]
# result = totalHammingDistance(nums)
# print(result)

def minPartitions(n: str) -> int:
	return max(n)
# n = "27346209830709182346"
# result = minPartitions(n)
# print(result)

def longestOnes(A, K) -> int:
	import collections
	d = collections.defaultdict(int)
	start, ans = 0, 0
	for i in range(len(A)):
		d[A[i]] += 1
		if i - start + 1 - d[1] > K:
			d[A[start]] -= 1
			start += 1
	ans = max(ans, i-start+1)

def longestOnes(A, K) -> int:

	if set(A) == 0:
		return [K if K <= len(A) else len(A)]
	if set(A) == 1:
		return len(A)
	if K == len(A):
		return K

	sw = []
	global_max = 0
	num_zeros = 0
	i = 0
	max_len = len(A)

	while i < max_len and len(sw) < max_len:
		if A[i] == 1:
			sw.append(A[i])
			i += 1
		elif K == 0:
			i += 1
			global_max = max(global_max, len(sw))
			sw = []
			continue
		elif num_zeros < K:
			num_zeros += 1
			sw.append(A[i])
			i += 1
		elif num_zeros >= K:
			global_max = max(global_max, len(sw))
			print(len(sw))
			j = 0
			while sw[j] != 0:
				sw = sw[1:]
			else:
				sw = sw[1:]
				num_zeros -= 1
	return max(global_max, len(sw))

# A = [1,1,1,0,0,0,1,1,1,1,0]
# K = 2
# A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
# K = 3
# A = [0,0,1,1,1,0,0]
# K = 0
# A = [1,1,1,0,0,0,1,1,1,1]
# K = 0
# result = longestOnes(A, K)
# print(result)


def subarraySum(nums, k: int) -> int:

	from collections import defaultdict
	d = defaultdict(int)
	local_sum = 0
	for i in range(len(nums)):
		local_sum += nums[i]
	return d[k]

# nums = [1,1,1]
# k = 2
# nums = [1,2,3]
# k = 3
# nums = [1,2,1,2,1]
# k = 3
# nums = [-1,-1,1]
# k = 0
# result = subarraySum(nums, k)
# print(result)

def maxVowels(s: str, k: int) -> int:

	s = list(s)
	i = 0
	max_len = len(s)
	global_max = 0
	local_max = 0
	start = 0
	while i < max_len:
		if i - start < k:
			if s[i] in ['a', 'e', 'i', 'o', 'u']:
				local_max += 1
				global_max = max(global_max, local_max)
			i += 1
		else:
			if s[start] in ['a', 'e', 'i', 'o', 'u']:
				local_max -= 1
			start += 1
	return global_max

# s = "abciiidef"
# # k = 3
# s = "leetcode"
# k = 3
# s = "tryhard"
# k = 4
# result = maxVowels(s, k)
# print(result)

def maxTurbulenceSize(arr) -> int:
	pass

# arr = [9,4,2,10,7,8,8,1,9]
# arr = [4,8,12,16]
#arr = [100]
# # arr = [2,0,2,4,2,5,0,1,2,3]
# result = maxTurbulenceSize(arr)
# print(result)


def repeatedSubstringPattern(s: str) -> bool:
	corrupt_first = s[1:]
	corrupt_last = s[:-1]
	return s in corrupt_first + corrupt_last

# s =  "abab"
# result = repeatedSubstringPattern(s)
# print(result)

def xorQueries(arr, queries):

	from collections import defaultdict
	d = defaultdict()
	d[0] = arr[0]
	for i in range(1, len(arr)):
		d[i] = arr[i] ^ d[i-1]

	result = []
	for query in queries:
		if query[0] == 0:
			result.append(d[query[1]])
		else:
			x, y = query
			print(x, y)
			result.append(d[y] ^ d[x-1])
	return result

# arr = [1,3,4,8]
# queries = [[0,1],[1,2],[0,3],[3,3]]
# result = xorQueries(arr, queries)
# print(result)

def maxArea(height):
	pass
# height = [1,8,6,2,5,4,8,3,7]
# result = maxArea(height)
# print(result)


def totalHammingDistance(nums) -> int:
	pass
# nums = [4, 14, 2]
# result = totalHammingDistance(nums)
# print(result)


class CombinationIterator:
	from itertools import combinations

	def __init__(self, characters: str, combinationLength: int):
		self.comb = combinations(characters, combinationLength)
		self.l = []
		for c in self.comb:
			self.l.append(c)
		self.size = len(self.l)
		self.ptr = 0

	def next(self) -> str:
		try:
			result = self.l[self.ptr]
			self.ptr += 1
			return result
		except IndexError:
			return ""

	def hasNext(self) -> bool:
		return self.ptr < self.size

# ci = CombinationIterator("abc", 2)
# result = ci.next()
# print(result)
# result = ci.hasNext()
# print(result)
# result = ci.next()
# print(result)
# result = ci.hasNext()
# print(result)
# result = ci.next()
# print(result)
# result = ci.hasNext()
# print(result)

def combinationSum3(k: int, n: int):
	import copy
	result = []
	def dfs(n, i, tmp, l):
		for j in range(i+1, 10):
			if n - j >= 0 and len(tmp) < k:
				tmp.append(j)
				dfs(n-j, j, tmp, l)
				if sum(tmp) == l and len(tmp) == k:
					result.append(copy.deepcopy(tmp))
				tmp.pop()

	dfs(n, 0, [], n)
	print(result)

# k = 5
# n = 30
# k = 3
# n = 7
# k = 3
# n = 9
# k = 2
# n = 18
# result = combinationSum3(k, n)
# print(result)

def combine(n: int, k: int):

	import copy
	result = []
	def dfs(i, counter, tmp):
		for j in range(i+1, n+1):
			if k - counter > 0:
				tmp.append(j)
				dfs(j, counter+1, tmp)
				if len(tmp) == k:
					result.append(copy.deepcopy(tmp))
				tmp.pop()
				counter -= 1

	dfs(0, 0, [])
	print(result)
# n = 4
# k = 2
# combine(n, k)

def parse_input(c):
	c = c.replace('i', 'j')
	c = list(c)
	i = len(c) - 1
	neg = False
	while i >= 0:
		if c[i] == '+':
			if neg:
				c.remove('+')
			break
		elif c[i] == '-':
			neg = True
		i -= 1

	if c[0] == '+':
		c.insert(0, '0')
	return "".join(c)

def parse_output(c):
	c = list(str(c))
	while len(c) > 1:
		if c[-1] == '0':
			c = c[:-1]
		elif c[-1] == '.':
			c = c[:-1]
			break
	return "".join(c)

def complexNumberMultiply(a: str, b: str) -> str:
	a = parse_input(a)
	b = parse_input(b)
	c = complex(a) * complex(b)
	r = c.real
	i = c.imag
	r = parse_output(r)
	i = parse_output(i)
	result = "0" if r[0] == "" else r + "+" + i + "i"
	return result
# a = "1+1i"
# b = "1+1i"
# a = "1+-1i"
# b = "1+-1i"
# a = "78+-76i"
# b = "-86+72i"
# a = "20+22i"
# b = "-18+-10i"
# result = complexNumberMultiply(a, b)
# print(result)

def combinationSum2(candidates, target: int):
	if sum(candidates < target):
		return []

	import copy
	def dfs(i, candidates, target, s, tmp=[]):
		for j in range(i+1, len(candidates)):
			if sum(tmp) < target:
				tmp.append(candidates[j])
				dfs(j, candidates, target, s, tmp)
				if sum(tmp) == target:
					s.add(copy.deepcopy(tuple(sorted(tmp))))
				tmp.pop()

	s = set()
	result = []
	dfs(-1, candidates, target, s)
	for record in s:
		result.append(list(record))
	return result

# candidates =  [10,1,2,7,6,1,5]
# target = 8
# candidates = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# target = 27
# result = combinationSum2(candidates, target)
# print(result)

def frequencySort(s: str) -> str:
	c = Counter(s)
	c = dict(sorted(c.items(), key=lambda kv: kv[1], reverse=True))
	visited = set()
	sorted_keys = []
	for k1, v1 in c.items():
		if visited.__contains__(k1):
			continue
		tmp = [k1]
		visited.add(k1)
		for k2, v2 in c.items():
			if v1 == v2 and k1 != k2:
				tmp.append(k2)
				visited.add(k2)
		sorted_keys.extend(sorted(tmp))

	result = []
	for key in sorted_keys:
		for _ in range(c[key]):
			result.append(key)
	return "".join(result)

# s = "tree"
# s = "Aabb"
# result = frequencySort(s)
# print(result)

def subarraySum(nums, k: int) -> int:
	pass

class Solution:

	import copy

	def __init__(self, nums):
		self.nums = nums
		self.c = copy.deepcopy(self.nums)

	def reset(self):
		"""
		Resets the array to its original configuration and return it.
		"""
		return self.nums

	def shuffle(self):
		"""
		Returns a random shuffling of the array.
		"""
		import random
		random.shuffle(self.c)
		return self.c

# nums = [1, 2, 3]
# s = Solution(nums)
# result = s.shuffle()
# print(result)
#
# result = s.reset()
# print(result)
#
# result = s.shuffle()
# print(result)

def combinationSum(candidates, target: int):


	def dfs(i, candidates, target, result, tmp=[]):
		for j in range(i, len(candidates)):
			if sum(tmp) < target:
				tmp.append(candidates[j])
				dfs(j, candidates, target, result, tmp)
				if sum(tmp) == target:
					result.append(copy.deepcopy(tmp))
				tmp.pop()


	result = []
	candidates = sorted(candidates)
	dfs(0, candidates, target, result)
	return result

# candidates = [2,3,6,7]
# target = 7
# candidates = [2,3,5]
# target = 8
# result = combinationSum(candidates, target)
# print(result)

def permute(nums):

	def dfs(i, nums, result, tmp=[]):
		for i in range(nums):
			tmp.append(i)
		dfs(nums, result)

	result = []
	dfs(0, nums, result)
	return result

def threeSum(nums):

	from collections import defaultdict
	import copy

	def dfs(i, nums, d, result, pattern=[]):

		if len(pattern) == 3:
			key = tuple(sorted(pattern))
			if d.__contains__(key):
				return
			d.add(key)
			if sum(key) == 0:
				result.append(copy.deepcopy(key))
			return

		for j in range(i, len(nums)):
			if len(pattern) < 3:
				pattern.append(nums[j])
				dfs(j+1, nums, d, result, pattern)
				pattern.pop()

	nums = sorted(nums)
	print(nums)
	d = set()
	result = []
	dfs(0, nums, d, result)
	return result

# nums = [-1,0,1,2,-1,-4]
# nums = [-4,-5,-6,3,11,-13,3,14,1,-10,11,6,8,9,-6,-9,6,3,-15,-8,0,5,6,-8,14,-11,0,2,14,-15,14,-13,-5,-15,5,13,-13,-6,13,-4,-1,1,-13,14,-13,-12,-10,9,6,12,8,14,-5,-8,-9,-6,-4,-2,3,-5,-2,-6,-2,1,-5,-12,-1,-11,-11,-11,0,-4,-2,-5,-5,0,-2,-7,-14,-10,-10,10,-11,-8,-13,-13,1,-2,-7,11,8,6,-9,-9,14,1,-13,-9,-3,-9,-5,13,2,8,-7,13,-14,6,-9,-13,3,-12]
# nums = [-1,0,1,2,-1,-4,-2,-3,3,0,4]
# nums = [-10,-11,13,0,-11,9,1,-6,-1,-12,10,-9,0,-15,-13,4,-13,-1,-12,2,-11,-6,10,2,-6,6,-8,-12,11,-2,1,9,2,-1,-14,-1,-6,-6,0,0,-3,-4,-2,4,-12,-8,-7,-10,6,-11,-1,2,-2,-14,-10,7,7,-3,10,-4,3,-11,-10,12,3,13,-4,4,-8,4,-11,-4,-15,-6,-15,-12,1,-15,-15,14,-11,-8,2,-6,-7,-1,-14,-14,11,4,-3,-1,8,-6,-3,-12,-8,0,8,-1,11,4,2,11,14,2,6,-8,-6,-1,-8,-1,6]
# result = threeSum(nums)
# print(result)

def reverseWords(s: str) -> str:
	import re
	s = s.lstrip(' ')
	s = s.rstrip(' ')
	result = re.split('[\\s]+', s)
	print(s)
	return ' '.join(result[::-1])

# s = "the sky is blue"
# s = "  hello world  "
# s = "a good   example"
# result = reverseWords(s)
# print(result)

def sortColors(nums) -> None:
	c = Counter(nums)
	c = dict(sorted(c.items(), key=lambda kv: (kv[0], kv[1])))
	i = 0
	print(c)
	for k, v in c.items():
		for _ in range(v):
			nums[i] = k
			i += 1
	return nums
# nums = [2,0,2,1,1,0]
# result = sortColors(nums)
# print(result)

def subsets(nums):

	def dfs(i, nums, tmp, result):
		for j in range(i, len(nums)):
			tmp.append(nums[j])
			dfs(j+1, nums, tmp, result)
			result.add(tuple(tmp))
			tmp.pop()

	result = set()
	dfs(0, nums, [], result)
	result.add(tuple())
	return result

# nums = [1,2,3]
# result = subsets(nums)
# print(result)

def permute(nums):
	from itertools import permutations
	p = permutations(nums, len(nums))
	result = []
	for record in p:
		result.append(record)
	return result

# nums = [1, 2, 3]
# result = permute(nums)
# print(result)

def countVowelStrings(n: int) -> int:
	if n < 1:
		return 0
	def dfs(n, i, chr='a', pattern=['a', 'e', 'i', 'o', 'u']):
		count = 0
		if n == 1:
			return 5 - pattern.index(chr)
		for j in range(i, len(pattern)):
			count += dfs(n-1, j, pattern[j])
		return count
	count = dfs(n, 0)
	return count

# n = 33
# result = countVowelStrings(n)
# print(result)

def grayCodeRecursive(n: int):

	def dfs(n: int):
		if n == 0:
			return ["0"]
		if n == 1:
			return ["0", "1"]

		result = dfs(n-1)
		ans = []
		for i in range(len(result)):
			s = "0" + result[i]
			ans = ans + [s]

		for i in range(len(result)-1, -1, -1):
			s = "1" + result[i]
			ans = ans + [s]

		return ans

	result = dfs(n)
	nums = [int(i, 2) for i in result]
	return nums

def sequentialDigits(low: int, high: int):
	def dfs(i, result, tmp=[]):
		tmp.append(str(i))
		if low < int("".join(tmp)) < high:
			result.append("".join(tmp))
		dfs(i+1, result, tmp)
		tmp = tmp[1:]

	result = []
	dfs(1, result)
	return result

# low = 57
# high = 610
# result = sequentialDigits(low, high)
# print(result)

def subarraySum(nums, k: int) -> int:
	from collections import defaultdict
	import numpy as np
	nums = np.cumsum(nums)
	total = nums[-1]
	d = defaultdict(int)
	count = 0
	d[0] = 1
	for i in nums:
		if d.__contains__(i-k):
			count += d[i-k]
		d[i] += 1
	return count

def subarraySum(nums, k: int) -> int:

	from collections import Counter
	import numpy as np
	c = Counter(nums)
	nums = np.cumsum(nums)
	c.update(nums[1:])
	stride = 1
	while len(nums) - stride > 0:
		for i in range(1, len(nums)-stride):
			tmp = nums[i+stride] - nums[i-1]
			if tmp == k:
				c[tmp] += 1
		stride += 1
	return c[k]

# nums = [1,1,1]
# k = 2
# nums = [1,2,3]
# k = 3
# nums = [217,-315,-999,-537,116,46,971,31,-978,-796,-613,80,-668,952,505,306,-405,884,435,-795,89,539,170,-963,360,-27,699,951,440,-163,-996,820,-548,400,-898,803,-771,-263,9,-201,661,881,-152,-173,966,406,-944,95,66,371,-202,-950,-532,-397,-662,-403,-21,-924,583,535,53,784,-75,750,924,-108,-95,-676,-525,278,699,915,908,-454,295,343,782,-478,699,801,-716,-322,555,583,-885,-437,-982,208,-486,618,1,-58,451,-965,-887,911,591,-639,-362,844,-726,-37,719,200,-943,-61,-211,636,963,-388,111,34,-160,-605,-458,569,26,-91,472,221,661,-172,253,-580,686,731,56,373,-355,859,-156,86,914,171,-654,-136,-523,-7,748,771,82,211,631,-796,-259,-29,552,-243,-842,-60,423,324,-383,281,-806,692,-152,-356,-320,-163,-156,142,689,-222,-156,-860,-708,235,834,-336,-329,-844,941,510,325,969,642,-498,887,671,931,750,-127,936,527,-990,-587,-337,-308,48,874,-276,-204,-28,173,970,-983,-971,236,541,-51,-735,-703,-959,-187,-628,-694,137,518,-491,-771,58,428,-354,-913,-23,-391,52,-569,-299,-632,-91,985,-32,661,-65,917,913,711,-361,259,575,233,-30,923,-421,793,31,358,-234,111,-302,200,869,441,-883,-224,-555,585,643,-891,317,97,-809,-938,145,754,448,-548,320,-49,813,-531,-816,451,-930,-338,-558,-583,-188,491,-668,-259,-518,85,225,881,-818,-669,-561,-726,556,625,456,894,-758,198,-912,382,857,-674,89,588,-396,197,-180,-745,-475,-19,-585,30,-717,-711,230,-698,-479,856,792,-318,-377,-560,704,868,73,365,826,154,637,772,868,-625,-52,-133,328,-862,-388,-465,-108,821,-837,-360,332,63,-487,-901,-95,-826,-354,299,-583,-756,969,940,-61,-740,769,181,-250,-920,-35,447,547,-379,-959,-139,339,-405,553,-728,110,256,62,-407,-768,416,-695,-532,-835,-146,-40,-562,-758,204,-143,90,421,607,681,-221,-691,-544,-643,-105,609,525,422,313,-553,795,-251,215,-290,529,366,-610,-128,66,-264,-105,595,-515,592,828,-31,137,-764,-594,795,610,-731,-604,742,-605,-160,613,429,-963,514,-479,-13,193,64,157,-167,27,96,275,121,-234,511,586,-923,-18,-733,674,-391,-715,-917,-48,106,966,-146,-570,850,-780,563,-892,879,175,-902,-431,-593,-186,650,-466,624,-732,-257,728,-557,533,318,308,-914,489,-526,840,-120,268,583,612,-853,793,451,520,-589,661,-224,-594,261,760,-868,-962,931,290,-562,-315,122,-968,-149,-719,482,929,597,228,-75,-675,-136,51,191,-685,254,612,-271,-987,933,15,418,247,-394,723,462,-615,949,131,761,-977,-788,-816,710,-982,-219,249,484,125,-318,-821,403,597,783,-15,-480,-716,-618,534,-788,970,-88,-993,-353,-871,-938,919,-339,-657,-154,-180,-555,-525,-443,-408,-748,379,836,-404,215,-8,616,300,-250,-836,-689,-299,872,570,-370,944,52,-894,210,353,-495,501,-660,-179,283,-977,-725,85,-775,38,-809,-328,-265,151,131,703,-817,397,542,299,576,-368,782,-439,-649,207,964,208,-768,929,-120,891,-31,-291,349,-601,288,449,521,130,317,200,-417,-717,34,628,283,-947,-239,-432,-455,-234,48,-44,-59,-898,287,796,701,-612,-523,923,876,854,123,893,890,656,522,477,591,323,834,744,-16,-764,177,517,613,633,-122,-585,-813,401,375,-122,-980,-554,-393,-816,884,227,881,894,-596,-693,450,311,78,-357,-93,-385,-174,592,-258,724,-973,-284,814,-308,-629,-138,-934,-380,793,831,764,849,-332,333,-351,-110,313,-647,271,-124,55,-231,-138,-231,31,-538,-826,-613,-103,-506,-109,-296,-949,298,549,-628,-332,623,-433,498,229,-324,322,-993,-880,139,-685,483,-901,-843,558,-200,-325,360,-805,122,989,757,-743,746,961,-771,941,344,726,918,59,646,824,-427,933,443,353,-517,144,-503,848,-937,212,702,493,613,231,-24,146,239,-617,-20,53,841,967,572,235,-167,307,-258,728,-273,57,99,898,66,-470,-163,5,164,796,461,-986,635,-372,245,170,663,866,-661,-526,650,-762,-965,352,88,-435,922,-963,-847,-886,-4,-490,-934,-18,-275,-67,802,33,-746,-647,465,945,899,528,197,473,441,355,-71,-20,668,986,889,-195,-831,-856,390,-911,-911,714,-456,941,735,98,-328,-455,-597,-282,516,-737,-554,-753,-951,924,816,528,21,217,-908,-380,-50,773,280,-820,-961,171,-981,144,-542,355,677,44,246,857,-316,351,777,-460,440,361,254,34,652,-362,-971,-860,762,-692,-949,-318,164,546,-768,862,-213,-80,727,698,646,209,394,735,-763,210,323,-56,303,434,-153,693,925,4,203,3,123,556,-377,-697,-793,-267,-341,-562,900,-516,-455,-609,-645,-753,270,-605,-215,474,800,818,507,591,163,705,-982,-837,-889,-390,-139,-575,-47,451,-204,-699,68,567,-486,-839,-107,-739,847,717,-706,-194,-849,-574,-254,423,957,-360,-176,133,640,609,-997,-179,-975,-453,-571,445,-161,-213,468,927,175,120,-68,-170,-506,78,-872,-412,-99,617,776,-984,-293,912,-497,-772,187,-631,196,-105,-103,-583,776,320,-585,729,304,251,-697,603,-791,126,-529,516,606,91,409,670,-462,643,988,-181,-51,-550,-984,624,607,-543,-514,-464,-901,87,-340,437,977,309,134,711,790,-467,32,567,-590,535,675,-689,-197,20,-819,-282,-299,-873,325,-210,-524,-222,82,759,-88,-296,-567,930,259,824,177,-26,-289,331,-907,845,-587,993,143,628,-166,-900,-31,881,57,-796,637,-385,351,175,68,-400,657,-168,837,237,-600,-157,-909,742,-222,617,729,16,869,712,-338,-252,-31,155,-227,685,34,367,-897,-393,-701,-377,-666,569,-400,45,-545,-464,555,795,131,-30,-748,674,-20,-84,973,-148,401,-565,-23,173,267,618,-208,-710,527,-568,-917,-749,424,657,373,950,-858,577,864,120,171,-607,30,262,-409,194,432,-477,-351,-963,492,197,-299,570,-199,573,347,-393,-455,-50,-595,760,-604,211,892,421,576,-866,-202,-785,-298,625,145,186,-105,-300,926,831,-109,-367,-70,187,883,-721,-685,-867,599,858,985,-624,109,-914,145,-561,928,-496,251,289,110,-106,907,-644,-573,834,263,240,-510,-504,152,66,10,-643,-388,-979,77,738,-230,-317,905,430,-868,-501,806,-346,417,446,656,634,363,441,-265,-946,687,-526,523,3,-972,-261,-781,81,-746,-877,936,201,-308,474,123,803,-213,898,-767,166,953,775,621,-89,819,683,96,-19,-704,654,-351,-609,301,-293,511,37,-448,-682,388,55,186,129,-272,-690,173,-29,399,156,192,198,873,597,904,-130,-406,734,-30,-496,-247,-839,-334,196,119,-782,-696,321,670,-863,-994,-133,-910,36,-697,-559,260,261,288,234,-373,570,216,-427,852,566,-325,-358,320,-670,-781,788,607,520,-346,-909,-283,399,-282,86,786,-872,789,496,-770,83,-889,42,142,-509,44,-362,910,-458,471,-890,631,302,-972,-915,379,-236,607,-206,93,-453,-430,733,315,226,198,-236,-347,241,-422,-118,-92,482,370,670,533,-721,-697,393,97,-503,-319,-567,433,449,-309,-733,-400,-686,512,41,537,818,325,661,-848,956,-309,193,-964,459,43,634,-487,610,-785,145,142,-561,909,-301,239,-678,-768,-500,-146,-762,7,-993,-810,830,-833,475,115,-899,-491,97,-621,-130,-792,400,351,-217,-437,-880,-138,52,112,-578,235,-209,727,-391,811,398,-89,429,21,-499,552,-744,694,-498,15,-592,921,-591,-507,123,224,643,-817,-73,-961,731,700,508,52,-85,8,374,703,-358,-535,156,-663,802,427,-965,630,-479,-954,714,-610,781,524,-489,347,347,-735,894,211,397,464,-248,255,731,-666,-953,-341,-468,-95,557,138,181,-682,702,-902,120,872,-62,-157,-786,730,-597,-987,-795,-56,-194,829,-619,-711,-883,-806,112,-35,244,-638,180,891,-615,637,990,-85,353,-278,757,906,587,583,-373,629,-880,-29,439,971,541,876,-527,-358,803,981,272,-62,490,-468,382,-253,434,963,-648,-640,462,547,-334,-365,-318,578,-510,-180,740,884,-242,253,576,-503,891,-884,-67,754,960,-918,457,-814,778,842,955,-957,-499,536,-688,-87,-82,198,981,-884,572,-479,-300,447,-800,146,737,374,-989,-762,361,831,-678,753,646,110,-826,760,-623,-932,-300,613,475,975,168,914,641,607,-255,-596,875,-490,544,-674,150,708,-655,-433,-380,-593,-870,205,-934,-57,387,635,-193,102,-381,752,-515,458,492,-275,-83,-193,-887,-223,-167,291,-593,-34,317,-199,180,-237,-191,-831,-496,527,315,-980,883,109,298,-256,-879,417,48,-288,894,639,346,251,-115,-315,809,952,101,449,353,210,-63,-837,613,534,-414,-296,-114,173,682,345,-65,-922,-211,735,279,927,-338,-761,124,606,-424,-419,-899,575,235,-922,975,-236,459,790,832,62,710,18,643,-205,87,199,693,-887,650,-472,-738,-598,-643,-902,-432,859,841,65,26,-657,996,-694,481,122,-520,156,-582,-305,408,-292,735,-89,-134,684,-67,-676,-36,-700,164,818,179,-140,833,204,-955,-586,-740,140,-4,-94,351,318,-44,15,223,757,101,-854,-281,-71,-351,-877,958,729,-974,-719,-387,-38,640,942,63,435,-394,-754,264,-876,-61,736,-205,86,-115,-765,-151,820,-509,505,-329,-943,-143,-601,-518,990,-124,-362,55,973,522,-149,524,269,467,-36,-699,-223,-39,-348,-57,741,-261,191,821,728,-941,552,-688,382,7,939,68,-240,87,359,-530,-48,-782,-626,-989,408,260,-977,-910,535,626,200,-617,-928,72,842,-647,-747,-933,-850,-378,471,122,-907,171,356,44,-935,-971,329,-379,725,341,-26,-90,-739,258,-862,950,224,172,958,-373,511,739,727,155,-872,-303,947,-19,537,-5,169,532,238,548,773,186,967,-222,644,229,-503,395,21,-357,-666,863,-991,203,-138,-825,-938,996,-386,143,562,-985,23,127,-296,-178,643,635,803,-938,-171,-279,-55,-256,313,-751,103,502,193,-870,70,-893,-701,-482,-428,-72,-319,156,967,486,-750,-723,833,-538,604,575,-144,-921,440,575,-950,-125,-911,351,225,-133,777,-503,-754,648,-982,744,-873,322,738,288,13,-432,-120,-661,-268,226,715,836,8,706,-651,-400,169,-273,181,666,-807,876,-159,-116,969,844,-229,-883,-222,-21,-456,-361,-846,698,25,509,-699,88,-950,418,192,115,85,-443,-677,-788,-841,19,880,148,335,911,532,-897,-616,-735,-677,-486,445,-591,-767,-984,93,-466,912,968,225,501,-929,-303,-652,410,-506,525,452,-692,180,359,-938,84,-720,443,-779,454,358,-924,-769,385,-800,359,-685,66,-658,-681,-946,-917,-207,231,-64,-283,-324,-571,717,-88,181,-389,958,110,601,618,613,-695,-132,-944,763,-669,98,-699,410,-936,336,-358,-774,442,445,190,932,389,658,559,565,970,-275,-218,-474,-375,-863,-993,15,376,169,-371,-691,-912,-863,930,-23,-321,919,484,-900,-348,926,166,422,-514,-716,-795,-441,-224,-155,246,389,-639,-919,-472,506,-664,244,-833,-942,-418,118,-535,-6,-256,-526,-911,-293,367,-563,751,-457,679,596,-265,83,-419,712,159,-547,822,-779,540,593,731,-534,494,232,-797,622,738,278,936,103,-355,-506,566,172,-817,380,-245,270,-872,-241,929,-871,88,298,476,738,-630,-293,-783,-132,-654,318,-881,-145,-219,-659,389,13,919,697,-94,-196,574,1,-347,189,-497,246,-688,-970,-415,-420,419,735,-781,-438,-592,-726,964,737,-588,-454,514,769,-302,-214,-562,-538,-393,-153,592,-365,-395,-98,-629,529,739,633,270,-784,761,-637,229,54,-294,-740,561,-177,-496,819,301,267,-657,-729,374,-506,278,-456,-888,97,501,-34,-281,-939,766,-204,-107,-871,-440,815,-7,-501,-489,917,-107,-361,-741,-793,515,-117,-196,-181,266,171,718,213,451,428,16,-158,-450,-816,282,-77,-878,491,-266,-241,-17,-606,923,99,-858,89,-948,-787,-305,-451,-867,-723,380,-260,129,-97,364,82,-245,382,628,-35,-702,-900,354,-506,-991,207,968,46,645,20,688,-470,266,-67,992,-737,-122,-115,146,-58,752,418,-401,405,152,-487,-935,220,839,-83,-429,798,799,922,-917,-119,682,945,890,975,-12,428,-262,-862,-879,-78,-830,-414,933,669,-845,-347,-616,-708,290,-147,-856,1,-866,-466,455,548,-802,58,-12,545,217,-269,179,-220,-504,331,817,-367,278,-4,-423,234,800,-857,-14,310,-901,-513,-502,368,-486,-625,231,-79,-173,74,-706,-230,-332,873,-462,595,-249,-860,778,-321,-327,221,-231,735,-24,-76,739,980,411,-288,769,-174,738,-664,-574,52,-31,805,185,-246,-924,-833,-543,738,124,856,-434,-590,97,-929,854,516,637,547,-758,636,392,-418,-830,586,279,614,-335,525,100,-126,-232,779,-469,-975,951,686,121,-683,-674,129,-698,-143,672,-948,725,823,-237,440,283,576,919,-838,-738,-809,-52,31,-212,-996,79,-42,-74,245,-460,-423,-150,780,273,-267,67,-412,-260,580,218,-431,-489,784,923,64,782,-62,-67,-802,-639,279,65,962,576,281,-589,743,408,-148,268,80,-908,-406,891,-49,444,797,554,255,-575,989,837,851,-712,406,-588,857,613,-252,-416,256,479,434,-520,-277,-130,360,-426,194,297,893,632,713,430,-625,-951,-217,-319,-895,3,349,115,889,893,-123,-629,185,-231,577,-549,-900,-393,-922,76,-173,961,650,-723,342,835,34,-104,-965,971,-33,-565,-284,-835,214,961,845,-849,484,566,-573,-55,101,412,-552,904,138,-826,-88,618,941,132,467,680,55,703,468,808,428,-60,-143,-246,460,102,108,706,944,-960,825,547,-233,994,-239,-159,531,233,-165,843,-227,-773,-919,190,889,534,809,410,430,491,-111,729,-136,934,288,619,-221,975,-960,-475,341,552,-82,-881,900,-869,110,-240,-880,183,370,-963,-68,803,747,-545,-441,86,-447,-287,-747,875,-654,-447,-235,-329,369,516,201,-147,-942,-506,-306,143,-39,-775,-969,-278,-735,964,6,-23,618,877,860,-354,615,-683,306,-42,-526,535,-744,520,-121,-813,875,-987,-697,804,-953,-375,-959,983,704,42,-766,-946,-875,-666,-971,-260,12,350,-762,111,234,937,90,-899,-278,212,-239,-234,306,-489,124,816,-255,803,-324,509,714,-867,-813,-413,-691,-502,-221,-7,-573,532,633,-247,789,-462,790,684,658,-602,618,263,655,176,865,398,-751,28,-744,403,441,868,978,-402,440,806,-664,790,450,796,25,-497,83,-833,568,314,331,-834,815,806,811,-194,848,-34,332,-257,767,822,-863,-334,458,-660,779,-893,106,-441,-894,-782,719,241,616,944,516,666,493,-12,252,748,-736,694,-313,472,613,281,550,-40,845,-554,403,-952,-640,-350,39,-501,-954,-456,-658,457,-545,598,766,-683,287,535,793,-352,-186,-521,-483,578,-817,692,-359,-161,122,-800,-928,-342,992,-865,828,-71,587,-271,913,688,-47,141,697,55,587,-163,212,-754,637,-101,-901,-231,101,-755,587,53,601,383,975,-428,710,-864,-825,-719,-331,-1,-963,-569,-240,170,-385,-300,-44,-344,930,911,-595,867,172,498,-523,182,596,933,-383,483,-263,654,-230,-582,-49,601,-698,-599,-160,-85,-901,-907,760,-141,-897,624,945,-829,960,659,48,513,-335,-732,-358,478,457,154,340,-643,-627,-873,807,-80,945,844,-559,228,-25,-41,-479,-210,595,987,790,-341,-220,896,671,-426,-492,24,-941,830,31,-436,-334,31,-931,-439,-268,-483,-597,408,575,-579,770,771,-121,-689,-927,431,796,-642,604,-266,-164,711,163,192,824,-865,-399,-980,50,355,602,-775,-379,-364,-933,392,430,42,580,155,-572,-890,85,119,-500,-583,797,998,-778,265,-762,480,927,-696,132,450,824,244,-640,122,-846,456,-179,-255,-819,-623,-486,-968,526,-741,-872,-605,69,136,713,-27,-867,250,-893,292,872,784,466,-441,-365,320,965,109,843,-491,-942,254,-150,738,921,-727,929,-115,157,-294,32,303,-843,810,89,-350,952,-399,131,801,234,-360,179,-967,-202,-999,68,478,390,427,-465,801,-233,-733,268,106,489,140,838,-467,677,216,-816,824,948,921,-517,-849,212,-359,-426,-503,-215,259,-92,-578,-558,-130,-278,697,-868,385,306,-96,370,-325,-644,17,836,-149,579,146,134,-925,891,-776,554,44,-576,944,-155,564,-708,-344,-91,-518,-46,429,-549,-895,381,-324,-563,-304,-126,-81,3,670,668,-976,-873,-818,610,737,27,893,603,-311,342,-655,-97,-140,-200,-557,-354,290,983,-609,-319,-655,264,-639,415,649,-846,836,-79,-60,-570,-863,-677,-8,56,634,-503,-123,653,230,-315,-269,-171,-853,93,115,-757,676,579,-320,418,-72,-781,805,47,380,-644,-743,-986,-671,-539,226,-482,-498,376,133,-577,-385,438,865,-627,468,-349,-331,-608,596,815,890,-618,909,362,65,764,723,460,-332,172,-792,-7,-760,776,-425,-887,945,-422,-137,660,50,454,-591,379,457,660,627,-351,80,-919,-418,-16,41,-67,-373,148,-672,-617,657,355,832,-571,-114,918,-374,945,829,71,-6,-289,-841,-608,695,-248,532,608,-136,811,98,-182,54,-270,673,177,957,61,348,-72,-106,-875,781,-48,817,-624,235,26,284,-323,969,-506,-267,849,503,-75,-196,756,-330,-643,466,-725,77,743,-595,-273,821,799,685,260,709,-729,-195,-293,-878,515,551,-790,-121,-32,210,-301,13,620,407,239,341,-226,-713,368,157,-144,225,13,616,976,-312,855,293,-428,492,-288,416,-320,-958,-867,-914,719,350,139,-279,898,-929,101,-933,261,204,898,990,710,-57,-147,563,590,-464,636,421,-243,163,798,-800,-595,-158,-108,-880,911,305,-566,775,194,574,-543,33,509,852,743,250,-491,-336,-768,-798,-693,-402,737,748,16,780,-951,-351,331,-422,351,-943,696,503,-32,-220,578,-383,443,-618,660,627,384,-175,-849,94,-955,2,84,-882,-377,455,702,464,-538,944,-421,-333,-270,-485,787,-830,630,252,676,686,-789,632,-652,-92,629,-121,808,528,-724,78,-13,985,-857,-734,373,-968,-403,-137,-824,711,507,-128,-149,-516,676,-302,-86,138,-174,827,-249,536,-749,354,-245,-591,479,-27,-307,-67,643,518,-956,-491,59,-687,769,-460,-278,-557,492,-326,400,686,255,-665,546,-533,-748,468,877,-977,388,-575,399,-474,-544,-852,450,-363,-598,-958,-116,-632,58,-30,-80,577,624,-6,633,-150,412,955,-729,-17,313,894,278,-52,-697,763,889,669,996,-588,-340,114,-338,-366,-538,-756,-405,-533,470,642,795,-461,576,-173,857,96,-322,838,-12,-781,347,107,168,-158,450,932,367,913,-61,29,697,-198,716,367,529,71,479,-669,-121,914,-898,542,465,166,-712,84,-733,830,132,974,924,782,-207,753,-686,647,933,-436,-619,-830,895,89,-387,-937,192,539,-573,601,376,-899,-316,-951,223,-633,-557,354,245,-139,-233,35,-933,-103,-94,227,360,-26,-864,-253,-768,-793,688,414,-439,410,-950,201,572,205,-541,405,-434,223,-406,238,-127,688,577,-343,433,937,433,-246,-132,-450,180,-718,-870,-535,400,-286,408,364,70,860,939,337,704,335,-602,-145,-373,-461,-160,-810,-408,-226,799,-448,-994,-198,-464,212,-636,-996,967,-142,-245,776,707,497,107,884,-410,538,-437,546,-231,608,592,797,-494,-507,853,-302,748,225,-593,-360,737,-534,-546,-165,941,-167,367,3,763,-912,396,-908,-82,843,658,-560,264,-360,275,-598,-790,904,-623,-657,-899,524,-123,-48,-867,148,-523,-486,-260,-477,690,-33,-721,-793,336,42,965,702,294,807,-912,4,669,141,347,233,913,-623,168,-525,548,-97,-443,377,972,-237,36,-611,736,952,-83,754,703,834,-378,823,231,673,-147,-567,-636,-526,-110,980,808,511,648,-890,-112,-582,-332,-906,359,641,-229,311,405,-298,-634,74,-700,-505,-367,921,627,696,489,194,91,366,703,847,582,259,-754,727,715,13,-750,485,317,-886,478,272,371,-10,721,-824,-416,-387,43,966,-671,607,-693,67,-403,859,424,-686,452,953,-396,559,120,866,-788,-637,-461,-939,843,192,-967,462,-495,-990,-831,-338,86,-626,104,-435,765,-666,-264,-613,68,-668,-367,149,944,-543,580,-550,982,-624,117,938,-93,-979,247,-171,-717,994,998,146,-451,-265,-448,157,-228,-355,389,674,360,-580,360,-897,273,386,388,-814,759,-879,752,9,-603,-807,317,-445,363,41,-800,-647,-85,-74,-777,453,387,696,947,-957,-719,271,-599,510,710,576,-492,-211,-914,-262,-755,500,-485,-670,915,-739,-329,-226,-85,-640,-734,275,-802,-598,553,273,911,-85,350,73,-134,713,-988,378,985,422,-979,623,334,455,-997,603,184,868,356,947,686,384,936,-764,-779,-333,-870,-505,467,932,-401,152,-191,-736,978,771,-463,-558,-257,-742,216,513,-853,-720,57,-668,-921,-901,-340,-915,-893,-783,-908,655,-854,-844,686,-331,826,737,-862,-221,499,573,991,-23,847,-714,-267,356,680,-766,303,180,261,273,320,-651,160,791,-155,-530,-910,-525,327,-692,-144,-220,-671,-970,275,53,893,-289,545,-414,970,944,-46,-4,-356,-679,-620,713,-758,-495,221,92,-904,-577,149,177,799,-420,-92,-532,592,139,-205,-470,-169,281,-638,-540,-18,-250,441,-672,-386,684,-976,-309,-454,833,-500,-709,-994,-172,-264,-851,368,660,482,450,557,497,-666,706,-834,759,154,353,-364,778,497,882,37,-753,-342,-471,45,766,-231,484,100,808,-995,978,513,-771,503,590,550,864,-278,-387,-879,-452,70,-973,-657,-135,6,-761,516,-168,-401,176,-902,579,-640,-902,707,495,-735,-58,42,-814,140,-798,-875,300,-670,-655,430,278,-645,-22,638,-944,-848,954,-857,-274,219,-576,964,796,-80,782,-408,-702,341,-297,-23,-822,-812,739,850,-404,-428,-358,258,684,789,-500,555,889,264,-821,-27,225,-409,-165,127,-286,117,-96,-163,899,460,698,531,854,567,870,11,-374,-447,288,820,802,-482,-228,-133,627,-913,-676,-131,24,-708,990,175,594,148,-611,639,852,-605,-802,983,199,107,527,-201,-35,-410,-276,800,-678,168,657,472,-80,-279,-275,-688,-953,-864,603,408,-927,262,-626,-492,-544,57,-574,827,-589,-263,-362,-156,-157,-928,-74,-689,466,-323,-289,50,-198,962,-732,-501,963,637,-33,265,783,365,-331,691,-825,-311,183,130,-942,-233,-917,528,305,-6,-889,505,456,-107,-796,700,-226,619,-932,-561,80,-24,102,-755,-162,-212,-26,-284,-188,-659,-742,281,955,-118,101,432,-288,-523,650,889,483,41,698,790,-815,-15,-464,253,-724,-807,-719,-538,815,-364,-438,608,302,-987,-666,45,-740,408,-396,-560,-484,-165,-327,697,-46,-157,-142,-682,-599,-771,34,462,395,-142,21,-619,-6,-692,626,563,865,102,870,-794,-223,-457,824,330,884,-676,-569,625,-763,737,-795,372,667,-114,501,535,-47,302,773,322,-784,929,-636,-97,198,409,-380,146,122,720,-447,-656,-644,672,308,-861,422,-723,-368,-168,864,278,952,-333,63,389,-582,53,699,880,415,501,919,371,-166,632,820,-453,684,395,88,244,-988,514,-525,109,817,869,548,693,927,-75,109,754,-806,-196,-346,656,432,172,-391,701,-101,-433,914,-387,-117,383,-645,563,265,630,-142,307,-346,-896,-471,-366,-8,367,-783,-32,833,-928,167,-130,10,333,-88,329,-393,223,603,757,-469,423,-178,953,148,-794,181,-120,-512,850,92,859,-64,-44,437,-722,-518,-191,935,-831,580,682,787,921,321,107,341,184,388,389,-149,963,250,519,982,578,997,-176,506,-286,915,224,233,168,-834,-753,-874,666,-104,658,579,904,670,-547,-158,887,-862,611,-53,741,-388,376,189,896,-333,-989,-194,-304,-711,-94,816,870,-692,-219,-986,-518,-953,887,-293,476,-697,-556,442,-725,-995,146,-377,278,-169,-604,-132,421,260,18,-261,-969,-545,644,-166,-493,567,-280,-464,-351,20,-710,-146,-190,891,411,438,-483,460,938,-399,627,668,-345,638,132,-762,-600,-769,684,-953,782,88,248,-740,-195,-55,947,24,-940,-194,-818,-712,-884,-265,-890,474,-684,-375,-346,-304,312,-228,868,-900,620,602,-112,-380,-700,-381,824,-251,454,-460,608,635,-569,-969,963,-117,-499,-430,540,-334,-171,583,154,258,-813,382,-656,-785,-510,590,10,-815,796,258,-961,-441,-664,-800,133,-686,502,306,-133,-369,-768,-199,-760,441,605,-540,980,550,-470,659,373,-16,-733,-821,915,-696,-807,548,797,157,-995,-405,-586,895,-187,-565,450,-593,649,440,732,924,141,-682,-841,-461,-820,246,-407,824,-119,-234,-41,168,-257,-367,187,634,28,21,-742,-438,-588,-673,-685,-497,-515,100,-705,102,971,-981,-569,-281,858,792,641,-483,-137,441,416,905,-461,-695,843,780,-516,617,-279,-601,-404,976,825,310,-362,950,-181,380,288,265,584,-821,193,-618,-711,-216,900,-765,-305,843,302,-843,-44,-224,-949,-463,431,-304,311,8,451,623,247,-567,-234,-138,-670,235,826,-348,-405,-303,-865,829,15,-473,868,-689,785,162,447,807,-37,-952,-375,721,250,986,-506,808,119,-752,403,967,-48,128,398,736,582,-452,523,747,324,-707,-660,-43,-833,-914,197,440,-836,-601,318,357,-679,-885,712,627,-656,656,427,-183,-375,350,-592,840,825,-221,-745,638,786,-240,892,50,88,-145,89,-33,309,-733,-884,-228,-952,582,572,247,955,-62,-519,-270,-646,-332,373,-162,-465,-472,891,285,-156,936,413,-197,210,809,712,-357,829,1,-139,829,391,828,-279,688,-170,792,-316,-352,-709,176,-588,-912,828,-992,612,-135,-161,-686,558,510,-683,957,506,-365,-744,-204,824,-627,982,153,600,-372,-239,576,872,971,512,-886,-750,550,-811,287,-667,-973,-176,146,342,-993,-586,-369,798,-452,209,839,634,797,-118,-579,-53,-751,823,-31,-460,-437,618,-802,372,-960,842,-621,7,-56,796,412,-470,-166,393,143,747,211,622,-790,-182,-664,-271,-998,-670,-763,373,149,-79,724,-279,981,-794,299,449,483,195,78,730,922,-59,-277,117,-690,18,953,249,-200,-848,590,236,452,-22,-748,-270,825,-4,475,443,97,-982,796,111,81,994,-488,306,731,886,855,685,465,17,108,123,-338,823,662,-146,78,-506,-244,741,-989,656,-221,-96,919,223,414,121,539,478,-837,234,-582,-123,154,-53,-246,-485,201,-411,-20,-818,771,640,-365,139,-141,138,971,894,-751,-580,-160,283,-594,833,-579,556,-634,-791,403,-20,273,283,-992,553,610,705,-577,427,920,-127,-907,-283,-503,806,-393,673,-496,-843,-406,-782,91,655,-722,-257,46,-772,674,504,-846,-521,-252,-513,-719,311,353,-680,-374,158,-432,-76,689,-857,-918,-87,550,-352,-889,-881,725,-854,169,-360,953,-872,-821,-383,-940,-542,-381,56,-275,-243,-744,859,90,456,16,640,-172,-563,13,359,-704,-754,242,673,-169,293,-319,-848,-553,-374,645,-728,786,671,853,342,-208,598,-745,194,719,-753,936,497,96,686,-889,-168,404,-193,-989,908,46,-570,-492,673,-440,-239,-63,-98,-460,-254,-934,-612,-340,-431,-901,57,-671,315,-665,-17,-529,-54,88,731,663,-501,-122,-606,699,526,-460,620,679,-743,13,489,-511,280,-254,-22,626,750,-468,824,433,-768,775,-673,412,953,-330,894,-931,-772,552,-168,506,-957,791,832,448,786,159,673,-527,-895,-913,932,829,702,998,607,673,-720,490,-597,-934,-193,-260,-927,672,-15,621,76,-505,29,-766,309,746,-717,-162,717,39,234,386,252,549,-720,-796,-175,619,894,189,860,164,756,-606,403,17,996,281,-841,-561,-531,-808,-759,-339,385,-331,633,-113,-216,707,-636,155,304,325,199,-754,388,-280,816,772,-934,93,-27,-157,748,-924,-386,286,-801,792,-937,-568,194,940,507,887,-987,339,-125,417,-476,-379,923,283,-638,547,-792,542,248,936,-820,22,-807,780,770,47,-306,499,-588,834,530,34,913,115,-190,-885,-259,-707,-736,-925,-742,-703,-853,510,-897,-393,664,399,-907,316,74,-184,447,823,768,272,-27,58,738,-286,-682,-383,837,-812,-150,620,970,-357,-183,685,384,-110,-356,697,-996,242,-532,144,425,-989,133,239,-453,-629,-468,415,602,546,853,-427,-679,567,865,-741,-288,-312,443,-622,125,679,276,53,-587,23,953,-821,-739,39,-230,-879,793,-114,-311,12,204,-990,-317,-918,-839,-390,141,821,-233,-805,-444,623,131,-142,765,33,-442,-493,226,961,246,265,-603,-490,-790,-698,447,512,708,454,639,-581,831,-456,-266,847,372,103,200,-947,-683,632,531,902,-83,-349,-47,-196,-221,150,788,199,-19,-912,-855,777,631,-943,278,-544,155,634,-696,767,-814,958,-427,-663,233,-681,458,887,-317,166,977,-898,-841,-243,-663,254,744,325,560,277,-352,802,-116,393,-347,192,-198,846,196,60,-263,-90,47,-206,694,-275,-747,-325,-928,-572,480,704,-386,-724,806,-313,884,-550,-722,-261,-278,-62,-332,-983,977,-548,-939,-239,-829,-476,767,-381,-379,-36,-364,-985,-706,445,692,537,178,-538,58,-340,-495,-655,-374,-454,-287,-8,359,-220,21,546,-465,23,-151,934,111,-391,56,850,-930,-100,-558,-227,-749,933,604,939,438,-691,-662,19,75,344,972,-99,256,-596,626,-61,272,-592,-522,-798,-349,-830,925,-228,-504,-387,20,-54,-5,-774,-341,820,-726,-444,281,940,132,339,609,-59,-591,-953,435,-791,-861,-536,470,-921,965,149,794,-187,-416,571,350,-288,-328,841,-841,-84,-240,236,392,555,-9,546,-810,-806,325,-382,528,83,482,-21,567,940,349,-325,-919,-131,443,725,-842,-605,-164,-452,-980,742,-945,657,438,-925,-364,122,584,3,237,-6,422,-951,274,-210,838,417,450,866,17,879,989,-151,-418,892,989,-201,160,461,-527,88,466,-948,268,-20,135,-398,772,-133,400,-669,117,401,-265,700,-94,189,236,-327,992,-132,900,-35,955,-662,-478,-561,-908,-980,-340,596,-406,-41,-372,368,-335,160,710,-59,799,652,303,632,-289,-726,-237,819,297,-15,-162,-739,135,148,504,-161,-167,-881,-951,714,853,664,513,411,192,320,-905,-991,-300,812,-751,-300,334,801,-994,745,-705,436,696,409,935,916,-773,241,-386,47,5,930,-501,722,346,-994,-694,-709,76,576,498,179,557,-969,835,-712,846,-298,-761,112,-481,457,157,-745,833,70,178,328,-991,-659,42,869,641,-488,-485,227,-518,-163,-86,124,870,-613,-279,-78,125,78,415,-301,-731,-882,424,150,186,-759,998,-298,299,999,586,496,-951,-308,411,-806,-398,-153,212,-976,-38,-768,559,289,-872,-548,-317,706,153,-102,807,861,-583,-338,976,86,839,750,941,145,-982,649,49,-188,56,228,-44,251,-886,787,-781,364,487,-547,-650,-70,-518,156,985,-369,-748,973,-685,913,-322,164,-313,-557,-246,-769,260,542,-218,-780,312,827,744,-561,-242,-953,12,-750,-520,331,-124,-824,-927,81,-510,770,-851,10,227,-147,-220,96,-736,-439,604,-816,650,-59,914,-803,992,-749,-83,-828,-190,-452,419,734,342,-664,441,-111,-271,-132,273,-322,148,-907,372,-109,854,-774,-870,-214,-321,-217,-544,-619,-98,103,629,755,-611,-108,369,61,-456,1,815,-80,-765,-305,-399,308,881,-906,-853,-913,867,563,-214,-881,261,-347,-69,741,239,-542,398,969,702,327,450,308,-49,-827,595,-760,234,632,545,137,-590,-397,553,477,995,-403,-813,-899,829,-809,727,-426,-690,-258,-421,558,759,-811,-312,-921,800,-719,417,485,542,231,961,137,521,342,-390,76,970,-188,-396,318,-157,664,-880,-222,671,973,284,98,-334,-365,981,-976,625,878,649,-868,-106,659,630,-688,123,716,731,509,426,785,-896,-160,493,238,-736,976,-701,212,-628,-510,679,531,436,-47,-593,-32,-6,18,146,558,-428,670,-970,658,995,-195,690,917,221,237,118,-430,-105,824,910,-21,465,-232,-889,740,-355,104,158,964,865,341,-841,811,946,190,-277,118,669,774,-981,653,-201,903,-929,650,-660,992,-413,-649,-407,-416,45,-490,673,-930,-534,-257,532,575,632,650,-731,588,-654,962,849,864,467,-954,42,464,-109,-218,136,-201,-981,-160,462,312,-842,-502,-868,-936,-833,-829,601,-645,-121,823,647,-94,635,-846,35,169,413,-162,-174,-394,-95,-719,54,81,-410,-374,-778,858,568,-995,260,-210,-808,127,233,756,873,63,-29,649,-209,-853,419,-626,-768,-509,377,82,469,-205,807,-205,549,-99,159,313,-353,-94,915,-122,748,892,-85,653,-430,771,-95,-719,391,683,-731,-64,338,-789,-351,313,840,46,703,660,-437,-483,-494,-325,-137,-596,-231,565,-695,545,997,-649,89,-806,859,970,658,551,-334,843,-614,-48,-999,676,722,-252,-530,-563,937,186,449,-697,730,866,-855,-361,468,-735,-106,346,-489,957,521,426,-723,56,-842,-405,-643,-4,-162,253,-803,379,639,890,-336,341,491,870,-291,944,-350,786,-650,894,660,-250,665,85,79,-51,-270,701,-45,367,-197,257,-199,681,-154,-201,-339,512,-303,-286,305,192,-807,444,914,448,-719,-732,-743,-978,-363,863,-640,958,-551,-13,602,896,412,924,538,303,108,-555,212,-974,660,29,544,-583,-818,-652,534,-825,115,-440,-95,204,789,-186,686,691,-189,-135,561,-256,55,-7,443,228,826,-289,-558,-546,689,441,572,149,269,-655,-349,-379,-194,-702,-876,-264,-137,528,-503,-161,-515,-475,-56,662,-917,-747,558,395,-230,-486,406,-985,90,-198,-105,-498,818,113,71,125,604,615,347,-304,749,633,155,933,-151,986,-579,643,353,971,827,456,-411,-144,-791,-110,131,-159,987,481,-2,507,418,788,-466,182,520,862,-490,-580,-114,54,143,258,544,-491,855,1,-912,-578,132,-950,-798,710,-614,451,-377,763,-37,792,-844,179,-970,970,842,943,-485,773,578,922,3,194,-197,-402,-349,984,439,-606,-853,-874,-978,283,790,-831,628,-91,549,-944,975,39,5,-643,-52,434,-945,811,-680,518,217,640,771,591,-353,362,-569,50,504,-677,20,-881,981,423,-85,718,853,354,-870,-461,770,784,445,939,633,344,911,-854,906,382,-830,558,-37,799,-198,15,-756,177,-121,27,-931,-87,-932,-794,-309,583,977,192,-732,-376,398,-775,198,-316,-273,-996,622,724,921,769,-506,-546,175,-891,-351,544,867,-684,-435,505,716,936,341,89,740,456,-697,-99,395,-720,963,-859,-463,427,985,-919,821,625,503,603,-123,416,-500,497,361,-812,-882,568,52,661,-657,365,-514,-965,-463,-261,-452,821,-492,-751,157,-197,475,865,561,-988,-638,183,704,-903,-643,-390,94,-410,726,930,-64,-161,388,-490,480,202,-950,-210,284,-105,-497,-998,690,-684,431,-776,539,286,-850,-500,558,874,-115,956,-647,85,47,739,121,-391,839,-550,-266,-789,214,-1000,-440,639,-744,396,401,-281,693,-592,-598,957,-642,-220,-326,-129,812,47,-635,-172,21,-488,-835,162,567,642,880,338,60,-74,868,804,-633,-656,-870,328,-254,881,-581,-437,-103,-1,779,-781,123,195,851,239,313,892,800,46,-964,-535,-747,315,-745,-14,599,-51,-804,669,144,-463,364,748,864,552,513,-219,799,644,-985,115,197,-845,682,-762,376,355,-738,-700,-909,-140,89,-62,143,567,-602,-4,-152,-65,390,826,-29,381,770,-504,550,-913,-817,-881,-987,563,-452,-863,872,-861,-876,-921,352,-630,90,-974,868,639,717,-378,-949,-477,825,713,-439,-623,-901,-847,471,-720,577,493,541,535,252,-648,226,-668,-299,975,-661,-911,-922,814,-528,901,-771,594,-246,331,-17,-614,894,-751,-508,388,-903,-345,-811,60,-653,-707,176,635,936,-345,982,815,492,92,379,94,-567,-144,686,687,-398,-55,-280,147,909,-579,-274,96,-820,-673,-691,221,815,-976,177,632,148,883,543,75,143,-100,-762,-553,-64,-164,268,-317,-269,-390,510,-753,554,-468,783,-669,-855,-79,-881,-816,853,-418,-272,-213,821,402,2,190,19,-257,266,-546,476,-167,-205,104,-986,847,932,-780,-780,-527,639,-202,-247,485,-848,931,-903,-896,-772,284,567,-25,765,211,489,-256,-451,260,765,-537,131,-530,-171,-624,-56,-286,-743,272,-663,553,376,352,-615,408,613,-402,-39,227,-854,343,597,-61,85,-965,-569,-529,921,884,-877,9,-88,379,3,380,-551,-249,973,544,-894,-31,423,908,417,-63,-557,-639,456,687,224,733,187,623,-571,86,-603,274,-585,319,-930,-581,39,967,163,-278,-467,-534,124,-796,-20,-795,897,-178,813,356,-175,-593,-710,344,-913,282,337,252,-670,951,-421,725,45,724,-245,768,203,-890,-30,195,153,-673,625,104,619,-463,104,-149,393,-996,-754,-9,-915,159,-882,44,-729,727,750,-54,716,616,-713,-379,-346,289,496,-395,992,259,575,-480,27,337,-244,39,433,-537,-104,-567,627,-443,-200,489,-227,-596,617,729,-66,-781,796,544,-462,-803,-179,575,-608,249,331,985,-703,-836,965,417,804,881,360,-36,-27,133,-244,220,-908,-449,247,-697,-463,-499,-308,837,-475,-38,853,689,-792,602,632,-595,637,746,974,276,-493,-6,-64,862,695,729,-750,458,-408,383,501,-590,461,77,447,672,-680,-938,459,-884,-85,580,397,-621,478,-590,897,-947,-539,572,719,-570,684,265,-271,-588,350,-78,-741,787,-774,2,-373,-11,-975,-319,-990,680,830,88,-450,-360,599,-459,-499,-450,237,980,506,395,36,-53,379,-312,305,-500,-777,191,15,649,406,726,355,547,-786,542,203,-396,457,-670,628,423,480,45,637,335,258,610,-818,-336,-691,149,455,290,472,151,-714,-726,-650,692,433,865,518,635,-428,-885,976,588,841,54,259,-814,294,934,-765,923,332,542,665,-873,-254,-64,-412,-980,70,575,192,965,53,-653,953,-675,729,720,-231,-186,52,610,337,-824,-548,833,686,218,-561,-339,366,802,-501,12,-980,812,-769,-130,937,-46,968,874,-550,844,705,680,418,179,-981,370,-131,696,-762,-458,682,-214,-272,19,693,-849,568,334,98,181,-836,37,711,-367,342,668,-963,347,-195,-826,347,-487,-528,-430,-685,-862,-363,698,764,865,540,-127,-500,493,387,672,618,-85,-798,489,372,133,830,485,-378,-978,959,150,795,241,-174,-741,-756,26,527,-29,-623,987,215,462,-267,485,442,340,-90,-225,902,-485,-93,-426,-986,282,573,-782,-478,-138,-502,-819,-478,-334,-591,489,-231,-483,-504,750,353,884,183,929,-10,567,-163,-439,-981,-922,2,853,43,-291,456,413,69,758,-826,-689,357,448,208,-912,-161,474,-432,-448,409,85,986,267,254,603,-291,-30,-184,403,195,813,980,-692,-283,509,687,376,321,-772,-226,-431,349,-193,416,935,-867,-952,-835,-311,648,226,566,-210,-991,923,-774,-891,553,320,30,-715,-905,-402,-917,478,930,-428,-655,749,40,-172,273,607,-939,-106,-802,-130,755,575,135,603,-187,-285,-669,725,-855,889,337,410,-846,-602,-219,53,-295,929,374,756,-141,-525,-404,-25,-87,247,-836,59,-425,898,964,-37,971,96,539,730,843,-676,59,-810,387,-705,-274,-852,-490,629,695,-975,1,-781,-717,706,431,486,379,-30,102,-619,128,-576,102,-338,-74,-26,-388,990,-582,-282,-617,-250,158,916,-737,-316,-546,-888,-595,299,559,785,877,300,870,-946,533,-447,466,975,22,-8,93,-471,-97,-612,936,-644,762,-649,281,933,-920,-512,-166,-861,658,-859,169,-841,-703,56,-753,-213,294,113,931,-989,43,-699,-169,462,-606,-95,754,929,-619,-896,634,-176,-931,820,542,786,209,138,-949,-945,948,103,190,290,-603,491,-729,643,-997,472,216,-356,806,689,-371,996,837,-327,496,532,-746,-573,16,454,959,-803,-985,470,-157,-219,-460,-323,-92,-89,-800,535,-527,-319,-428,-335,-86,-613,-363,94,-929,297,775,632,873,58,108,940,-376,979,432,952,-864,-711,559,-373,-356,326,244,-192,858,-335,-263,517,-978,-257,-528,787,859,-351,834,-272,-535,-56,728,958,481,-164,931,-33,-875,-4,86,-449,-184,-159,577,787,147,-338,777,931,212,-85,-49,-405,-294,218,-503,-748,-471,425,102,-963,-839,81,-497,668,468,-989,-555,159,572,309,987,-552,102,-957,-442,-309,391,582,-978,-967,442,-271,14,-611,649,741,278,-805,-251,-866,-205,-136,153,-890,365,-247,-772,903,661,-570,423,672,496,-823,-660,-132,843,715,620,256,-133,174,893,-919,-943,-9,852,263,513,-961,294,-903,278,915,-847,488,310,-802,-625,-877,846,-235,-52,784,258,-499,618,109,-865,-387,-741,-304,-249,-409,719,-87,-940,557,930,-597,-972,435,764,-565,850,-829,-207,991,-718,563,-412,-886,-332,-219,95,787,-407,142,860,624,-738,-278,751,190,241,-396,918,-790,-486,-935,474,-439,-971,-589,289,-884,-535,-69,-920,-455,615,931,-47,-797,908,711,-176,-652,719,322,-639,673,89,959,-869,-613,382,-163,-856,945,139,653,-598,-392,209,-112,314,-790,733,-33,-809,811,-242,-808,653,-775,-682,189,-665,288,935,-809,-209,-244,300,-948,-897,665,-846,-499,-881,-82,203,-580,270,-850,693,-46,784,-289,957,-96,-527,753,-926,374,-274,817,205,360,-542,-163,449,-615,574,-821,-321,-135,-968,771,-462,293,-356,-420,941,-562,255,-634,-322,95,-654,-40,-249,99,580,891,-790,-573,-95,270,49,827,-557,-331,587,-122,302,-596,-426,886,620,-128,-43,162,-660,-523,783,-636,-272,712,-408,740,518,-179,288,-534,710,168,-820,817,504,426,-178,-510,-429,-339,-323,940,-963,414,88,529,509,463,307,209,713,-304,951,-473,-28,766,-933,-10,-712,-247,-592,-112,-274,616,-859,-782,20,-870,-953,-462,-794,-934,394,-301,740,769,-206,-308,-657,757,-946,-519,302,165,-204,-253,-538,-623,-344,-527,-349,961,-65,-951,-259,739,-3,39,139,-955,301,213,571,666,-202,-201,910,-283,475,-944,-311,-575,857,75,-288,205,692,-364,922,187,-973,-821,-643,-793,835,814,924,956,628,74,596,-291,406,46,181,47,-972,-400,43,974,210,625,358,73,-722,-272,-492,623,713,431,837,-496,471,-336,334,-531,596,-959,661,-308,66,-783,212,-257,321,-165,128,-539,-285,463,-795,-61,109,230,741,998,617,-149,424,742,-805,395,810,13,-485,891,715,-909,-764,-318,-785,161,-235,61,863,-230,-522,463,-337,-496,-234,422,-5,418,947,984,46,-786,-595,283,340,737,-232,-660,-940,248,89,-328,944,-528,136,-829,-795,362,-156,460,679,836,-951,-111,-959,36,-42,-65,484,688,570,-767,-988,300,220,-248,-928,-476,976,418,-680,-557,334,195,325,368,51,474,42,-89,221,768,349,496,968,76,-349,381,-438,-607,9,359,-399,991,417,63,768,783,-582,129,-226,283,-901,-777,538,666,-899,68,-483,-670,-578,-955,-780,784,-966,-634,316,-934,797,782,-580,-799,175,851,93,434,625,-629,-667,-101,320,-541,-796,-187,-732,873,-884,293,113,-150,-397,464,-783,770,524,421,-677,191,-293,772,-932,-841,-144,131,-514,-445,-600,-9,394,-866,954,340,-821,256,-847,-759,302,964,-164,538,-617,944,-457,-549,-586,715,-220,-116,-870,845,-36,-60,235,-25,388,-847,-178,416,-419,-177,643,-778,-87,647,-777,670,-935,-552,703,890,-421,-602,240,-745,-464,800,296,-672,-197,-175,-372,164,549,-645,970,483,76,809,769,303,-189,-105,-597,151,-178,-654,-676,-948,662,497,-228,627,309,-872,-592,714,-560,181,-980,573,900,870,-786,-736,989,354,725,-459,125,440,-826,-391,447,-544,-865,169,-595,-762,826,-846,439,-263,464,-287,-529,781,-440,-465,-26,-636,215,287,-880,-665,465,276,398,-83,643,590,223,-869,451,587,-901,-703,-896,157,-703,-415,-684,-240,-417,-378,894,807,455,122,-219,401,654,357,-897,653,-655,-723,-331,166,946,849,128,-271,-430,297,-626,-187,-434,94,758,24,117,-322,-941,561,558,-638,783,254,202,-966,-315,854,997,-849,-445,630,-588,771,708,-718,430,367,-165,-929,168,-358,-502,-108,811,34,228,293,-986,-971,862,-921,275,-631,-225,-812,970,829,293,71,254,-961,-395,-243,646,205,-537,-722,96,-102,708,468,975,392,-481,996,-480,617,193,581,-983,17,555,731,-330,-369,630,-438,852,724,545,-519,191,58,305,-629,-258,-939,891,-527,751,-538,-169,869,366,-439,137,518,-39,397,-132,715,-298,-849,725,852,267,-489,820,11,49,-213,-601,-771,463,-291,-198,-178,-642,748,-661,-109,-371,332,-624,954,934,-15,-752,92,-818,891,-823,465,788,-14,-18,-569,210,475,-103,917,-388,-839,234,768,-929,376,-397,296,635,-634,862,-903,-750,-258,-200,-759,-255,-75,419,58,-464,-364,907,256,123,-1,910,-932,113,-25,-948,144,173,320,505,-265,-662,-621,774,-230,-203,645,971,906,-710,-876,516,-719,675,-512,-736,752,240,271,-216,-910,508,440,376,-786,-797,973,756,-424,782,-618,-142,41,-837,906,-933,21,-953,71,42,316,-565,319,906,260,-217,-562,656,-322,671,241,810,743,90,506,-664,727,205,820,-299,115,-287,614,-809,-439,406,-80,-264,-850,-387,-10,931,-294,-573,-601,821,-723,986,477,315,-881,795,577,957,-427,-60,191,-273,318,522,190,-473,-24,-893,469,-694,760,-526,93,-8,-455,795,182,-655,998,-964,417,187,668,544,568,-293,972,837,-288,-238,681,-157,988,-342,379,-903,363,464,-714,866,-602,-389,-131,-447,-647,-523,536,-442,596,-553,-491,-418,768,33,-375,-62,-503,-303,-545,993,-710,-54,-788,-31,125,851,-209,-357,973,774,866,609,585,499,37,276,937,191,9,-427,636,404,-480,-159,860,-777,-697,182,-497,185,-730,-108,-448,-392,99,616,-78,-444,673,265,868,367,354,994,-755,762,-156,-202,884,-430,328,-347,974,922,573,139,677,-935,-428,838,-895,317,354,-675,-186,163,728,-413,213,833,66,-872,-355,531,9,-859,998,425,-659,-751,6,557,816,801,-595,-78,-422,632,-397,185,513,721,-238,624,253,375,688,-348,-974,679,284,-970,623,185,-914,-452,281,273,-532,-802,-450,-623,-265,-690,-550,70,-383,133,-791,-709,558,77,127,-440,909,897,-187,-917,-967,486,-415,-103,485,-655,-589,928,-709,113,225,-870,869,-608,156,-207,-451,559,336,-151,-640,478,347,809,-863,373,-487,-227,752,587,-456,-621,-182,-840,-531,805,243,-977,569,397,-594,-181,17,123,557,-436,-365,-993,968,-392,465,-310,425,-361,-128,409,821,670,-570,-122,932,-233,-271,-956,44,561,550,-851,-212,221,-871,108,-912,209,-368,-154,-921,160,429,790,329,162,-549,308,418,115,-265,-826,-469,742,-876,-273,662,562,686,-609,-941,-993,746,769,-410,-620,204,278,-305,319,57,209,-779,-834,-207,502,878,-347,-137,945,-129,9,58,165,330,-186,-806,-274,389,583,266,80,-284,-224,-841,978,-759,812,-1,340,-548,528,-910,-938,-918,708,-447,-213,128,-287,-257,329,-455,-831,86,415,-33,386,-972,874,51,348,-712,545,809,530,-687,-847,555,774,152,36,231,-894,-805,-421,291,-564,-376,377,121,222,337,412,723,801,-449,-780,957,-736,-169,723,984,-222,817,575,261,-27,598,550,791,18,-397,-867,-18,-874,-439,-3,-809,-498,-762,-317,672,939,-276,-386,-39,852,378,-127,926,602,-133,551,969,568,946,824,-541,-472,124,-852,-607,-414,698,491,171,-133,-255,-562,411,826,228,-7,309,-350,-225,-93,-950,992,356,-855,127,-431,532,-67,582,965,-46,946,-226,-626,288,-563,-726,-609,944,-474,-809,-696,-38,172,-973,-289,-562,-501,328,802,-117,656,-152,-731,-909,-657,908,807,-631,424,-914,712,445,-918,939,-307,-736,-501,759,-947,561,19,-523,-419,413,-760,19,95,-865,354,-386,-87,-90,-1000,-23,-561,-761,-455,324,-658,8,-905,811,-954,-827,259,381,515,-489,-666,-578,667,-115,696,-683,-938,495,-544,866,772,-687,-844,-751,607,-710,-62,-939,38,34,316,504,-549,-268,-561,-785,241,701,191,-383,-647,776,-302,-138,-710,-306,384,-929,164,316,-117,47,518,444,30,-194,70,-309,764,-412,845,-402,-523,896,744,327,-177,-16,198,48,329,-460,-332,-413,37,-650,-28,-400,837,-497,594,896,596,-305,108,-271,-118,736,-434,-461,23,347,-388,-27,-982,682,742,595,396,-778,-718,206,-495,-578,895,-474,568,764,-453,77,586,-917,-9,-640,-728,-787,51,115,-903,485,-280,-113,484,484,-939,322,-101,-826,551,-105,433,-914,686,247,855,-272,904,985,-260,563,-92,216,-887,779,-834,826,409,280,-153,843,-458,-43,-628,48,-116,871,-233,171,513,-963,149,-242,-596,-775,-982,368,-737,410,865,-917,-210,-586,169,-641,-390,215,-917,66,65,234,-451,-161,-591,-838,-727,-965,-102,405,632,-802,664,501,-331,-260,803,-479,584,-17,-338,684,-789,-785,958,-583,-26,820,58,465,953,-266,-246,-194,-87,-57,-588,-540,739,384,-361,811,918,955,301,-517,-573,133,985,-377,-601,-245,-987,-270,-216,231,977,-437,-452,433,680,-348,-232,-999,-266,539,469,48,120,831,335,364,477,-900,-157,-649,-429,-166,365,-627,-636,-817,792,-818,187,-368,512,926,291,-745,11,-287,846,212,-875,-907,-165,-205,71,827,-835,817,669,-883,328,-429,779,288,-246,-290,130,627,965,497,-684,887,-219,244,-743,387,55,344,164,-942,319,-810,831,4,-648,-739,-496,602,-365,103,498,350,-786,960,-968,-207,790,-861,-945,684,610,-256,632,991,825,-921,-402,-541,753,841,-910,-693,114,356,360,67,504,-823,970,574,-997,919,46,-312,570,562,307,-351,663,170,674,-366,703,-248,-42,-560,-417,988,320,-930,85,471,859,-676,-118,392,-501,-163,-45,-496,301,126,-682,-154,-866,-859,368,-592,-258,307,-280,564,-852,492,-148,995,293,470,526,-361,272,-273,-978,159,-510,336,-417,834,-254,-691,857,-671,-297,-270,730,-256,220,-216,45,-580,-382,-822,691,343,-654,313,-41,620,518,647,-206,-730,-929,578,622,1,818,411,426,-2,956,-187,180,772,305,-224,603,563,611,269,442,-323,933,-536,-579,898,345,642,-620,205,-117,621,748,252,-861,-290,-365,-949,227,-182,863,913,127,-523,529,-658,-99,496,-726,293,189,-579,-311,942,172,964,397,-636,502,-858,-777,467,819,679,-941,424,-534,-352,-430,434,-463,318,-209,-297,770,-762,-975,-943,-307,562,-21,-811,-191,-151,-434,62,993,-648,-553,490,496,-474,558,-810,770,-761,-208,367,-381,-893,-113,290,-139,-558,-530,574,-959,925,148,-304,-562,-894,901,-175,207,-857,852,954,369,-422,-145,550,503,-137,0,-434,-922,-288,210,-842,626,-518,281,-588,-143,723,-90,-207,-984,-64,-423,206,-599,-932,-536,-983,245,406,-170,949,642,43,-215,-59,-609,533,455,-73,-747,300,723,814,-733,698,961,978,200,-371,556,-114,-292,-87,837,340,-606,884,418,448,-391,-699,-530,-985,-213,722,-412,-224,-458,-602,486,358,579,886,291,-251,-966,-959,-706,363,-171,-427,-716,-322,499,-719,989,711,650,421,-562,-342,-482,790,59,-469,881,239,-318,954,-531,-538,-248,54,410,448,-911,-943,841,-930,736,-334,-837,884,348,388,-83,-231,390,-977,767,-171,-692,519,654,-915,303,382,248,58,-330,-343,422,-532,-102,-568,-282,762,-338,196,-743,-243,-614,-682,730,441,130,-237,800,223,949,497,919,26,609,815,-411,-973,-698,-880,821,-486,-173,183,-294,658,6,214,-321,766,-524,-93,-125,-389,311,-147,84,393,480,884,-959,-120,-848,83,-499,904,43,654,150,494,199,-681,516,-744,950,871,60,75,-705,893,545,485,-997,-52,820,417,-899,561,-593,-593,909,-942,649,-893,-990,972,-698,363,164,404,-240,-747,743,461,-82,292,176,290]
# k = -93
# nums = [1,-1,0]
# k = 0
# result = subarraySum(nums, k)
# print(result)

def groupAnagrams(strs):
	from collections import defaultdict
	d = dict()
	for s in strs:
		t = "".join(sorted(s))
		if not d.__contains__(t):
			d[t] = [s]
		else:
			d[t].append(s)
	return list(d.values())
# strs = ["eat","tea","tan","ate","nat","bat"]
# result = groupAnagrams(strs)
# print(result)

def permuteUnique(nums):
	from itertools import permutations
	p = permutations(nums, len(nums))
	result = set()
	[result.add(i) for i in p]
	return result

# nums = [1,1,2]
# nums = [1,2,3]
# result = permuteUnique(nums)
# print(result)

# Definition for singly-linked list.
class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next

def populated_list(l):
	head = ListNode(l[0])
	curr = head
	for i in range(1, len(l)):
		curr.next = ListNode(l[i])
		curr = curr.next
	return head

def print_list(curr):
	while curr is not None:
		print(curr.val)
		curr = curr.next

def rotateRight(head: ListNode, k: int):
	curr = head
	# Length of the list
	l = []
	while curr is not None:
		l.append(curr.val)
		curr = curr.next
	k = k % len(l)
	l = l[-k:] + l[:-k]
	head = ListNode(l[0])
	curr = head
	for i in range(1, len(l)):
		curr.next = ListNode(l[i])
		curr = curr.next
	return head

# l = [1,2,3,4,5]
# k = 2
# head = populated_list(l)
# head = rotateRight(head, k)
# print_list(head)

def subsetsWithDup(nums):
	def dfs(i, nums, result, tmp=[]):
		for j in range(i, len(nums)):
			tmp.append(nums[j])
			dfs(j+1, nums, result)
			result.add(tuple(tmp))
			tmp.pop()
	result = set()
	result.add(())
	dfs(0, nums, result)
	return result

# nums = [1,2,2]
# result = subsetsWithDup(nums)
# print(result)

def grayCode(n: int):
	return [int(x^x>> 1) for x in range(2 ** n)]

# result = grayCode(5)
# print(result)

def maxTurbulenceSize(arr) -> int:
	pass
# arr = [9,4,2,10,7,8,8,1,9]
# result = maxTurbulenceSize(arr)
# print(result)

def getHint(secret: str, guess: str) -> str:
	bulls = 0
	cows = 0
	i = 0
	secret = list(secret)
	guess = list(guess)
	while i < min(len(secret), len(guess)):
		if secret[i] == guess[i]:
			bulls += 1
			secret.remove(secret[i])
			guess.remove(guess[i])
		else:
			i += 1

	i = 0
	while i < len(secret):
		if secret[i] in guess:
			cows += 1
			pos = guess.index(secret[i])
			guess.remove(guess[pos])
			secret.remove(secret[i])
		else:
			i += 1

	return str(bulls)+"A"+str(cows)+"B"

# secret = "1807"
# guess = "7810"
# secret = "1123"
# guess = "0111"
# secret = "1122"
# guess = "2211"
# secret = "1123"
# guess = "0111"
# result = getHint(secret, guess)
# print(result)

def sumOfUnique(nums) -> int:
	from collections import Counter
	c = Counter(nums)
	count = 0
	for k, v in c.items():
		if v == 1:
			count += k
	return count

# nums = [1,2,3,2]
# result = sumOfUnique(nums)
# print(result)

def generateParenthesis(n: int):
	if n < 1:
		return []
	if n == 1:
		return ["()"]
	pass

def restoreIpAddresses(s: str):
	pass
	# def dfs(i, s, result, counter=0, tmp=[], tmp_str=""):
	# 	for j in range(i+1, len(s)):
	# 		if counter < 3:
	#
	# 			dfs(j, s, result, counter+1)
	# 		# tmp.append(s[j-1]) #or i
	# 		ans.append(s[j])
	# 		if 0 <= int("".join(ans)) <= 255:
	# 			print(ans)
	# 		elif j < len(s) - 1:
	# 			counter -= 1
	# 			return []
	#
	# result = []
	# dfs(0, s, result)
	#return result

# s = "25525511135"
# result = restoreIpAddresses(s)
# print(result)

def countSquares(matrix) -> int:

	import numpy as np
	matrix = np.array(matrix)
	row, col = np.shape(matrix)
	count = 0
	for k in range(1, max(row+1, col+1)):
		for i in range(0, row):
			for j in range(0, col):
				if not (i + k > row) and not (j + k > col):
					sub = matrix[i:i+k, j:j+k]
					if np.all(sub == 1):
						count += 1
	return count

# matrix = [[1,0,1],[1,1,0],[1,1,0]]
# matrix = [[0,1,1,1],[1,1,1,1],[0,1,1,1]]
# result = countSquares(matrix)
# print(result)

def readBinaryWatch(num: int):
	from itertools import zip_longest
	def dfs(i, last, n, minutes, result, counter=0):

		for j in range(i, last):
			if n > 0:
				counter += (2**j)
				dfs(j+1, last, n-1, minutes, result, counter)
				if n - 1 == 0:
					result += ['0' + str(counter) if counter < 10 and minutes else str(counter)]
				counter -= (2**j)

	hours, minutes = ['0'], ["00"]
	# result = set()
	# dfs(0, 4, num, 0, hours)
	# print(hours)
	# dfs(0, 6, num, 1, minutes)
	# print(minutes)
	result = set()
	# for i in range(0, num+1):
	dfs(0, 4, 1, 0, hours)
	dfs(0, 6, 1, 1, minutes)
	for hour in hours:
		for minute in minutes:
			result.add(hour+":"+minute)
	return result

# num = 2
# result = readBinaryWatch(num)
# print(result)

def minOperations(boxes: str):

	boxes = list(boxes)
	result = [0]*len(boxes)
	for i in range(0, len(boxes)):
		for j in range(0, len(boxes)):
			if j != i and boxes[j] == '1':
				result[i] += abs(i-j)
	return result

#boxes = "110"
# boxes = "001011"
# result = minOperations(boxes)
# print(result)

def numTeams(rating) -> int:

	def dfs(rating, i, tmp=[]):
		count = 0
		if len(tmp) == 3 and (tmp[0] < tmp[1] < tmp[2] or tmp[0] > tmp[1] > tmp[2]):
			return 1
		for j in range(i, len(rating)):
			if len(tmp) < 3:
				tmp.append(rating[j])
				count += dfs(rating, j+1)
				tmp.pop()
		return count
	return dfs(rating, 0)

#rating = [2,5,3,4,1]
# rating = [1,2,3,4]
# result = numTeams(rating)
# print(result)

def minOperations(n: int) -> int:
	target = (2 * sum(list(range(0, n))) + n) // n
	count = 0
	for i in range(n-1, -1, -1):
		if 2 * i + 1 <= target:
			break
		count += 2 * i + 1 - target
	return count

# n = 3
# # n = 5
# n = 13
# result = minOperations(n)
# print(result)