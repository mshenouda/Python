import numpy as np
import random
from collections import deque


def distanceBetweenBusStops(distance, start, destination):

	f_d = 1000
	if start > destination:
		tmp = start
		start = destination
		destination = tmp

	f_d = sum(distance[start:destination])
	b_d = distance[destination]
	for i in range(1, len(distance)):
		d = (destination + i) % len(distance)
		if d == start:
			break
		b_d += distance[d]
	return min(f_d, b_d)

def specialArray(nums):

	nums = sorted(nums)
	nums = list(filter(lambda x: not x == 0, nums))
	rejected  = set()
	while len(nums) > 0 and nums[0] > 0:
		nums = list(filter(lambda x: x >= nums[0], nums))
		if nums[0]-1 <= len(nums) <= nums[0] and nums[0]-1 not in rejected:
			return len(nums)
		rejected.add(nums[0])
		nums = list(filter(lambda x: x > nums[0], nums))
	return -1

def findJudge(N, trust):

	results = []
	for record in trust:
		if len(results) == 0:
			results.append(record)
		else:
			flag = False
			for result in results:
				if record[0] == result[-1]:
					result[-1] = record[-1]
					if not flag:
						flag = True
			if not flag:
				results.append(record)
	s1 = set()
	s2 = set()
	for result in results:
		s2.add(result[1])
		s1.add(result[0])
	print(s1, s2)
	# for record in trust:
	# 	s1.add(record[0])
	# print(s1, s2)
	# if not len(s2) == 1:
	# 	return -1
	# if not s1.intersection(s2) == set():
	# 	return -1
	# l = set(list(range(1, N+1)))
	# if not s1.union(s2) == l:
	# 	return -1
	# return s2.pop()

def largestSumAfterKNegations(A, K):
	A = sorted(A)
	if len(A) == 1 and A[0] < 0 and K % 2 == 1:
		return -A[0]
	if A[0] == 0 and K % 2 == 1:
		return sum(A)
	if A[0] == 0 and K % 2 == 0:
		return sum(A)
	if A[0] > 0 and K % 2 == 0:
		return sum(A)
	if A[0] > 0 and K % 2 == 1:
		return -A[0] + sum(A[1:])
	if A[0] < 0 :
		i = 0
		while K > 0:
			if A[i] < A[i+1]:
				pass

def summaryRanges(nums):

	result = []
	i = 0
	while i < len(nums):
		s = str(nums[i])
		flag = False
		while i < len(nums)-1 and nums[i] == nums[i+1] - 1:
			if not flag:
				flag = True
			i += 1
		if flag:
			s += '->' + str(nums[i])
		result.append(s)
		i += 1
	return result

def minMoves(nums):

	count = 0
	import numpy as np
	while len(np.unique(nums)) > 1:
		nums = sorted(nums)
		diff = max(nums) - min(nums)
		count += diff
		# if len(nums) > 1:
		nums[-2] += diff
		nums[0] += diff
		print(nums)
		# else:
		# 	nums[0] += diff
	return count


def findContentChildren(g, s):

	s = sorted(s, reverse=True)
	g = sorted(g, reverse=True)
	d = dict()
	for i in g:
		d[i] = 0

	for i in s:
		for j in g:
			if d[i] > 1:
				continue
			if i >= j:
				d[i] = 1
	v = d.values()
	return sum(list(v))

def prefixesDivBy5(A):

	s = ""
	result = []
	for i in A:
		s += str(i)
		d = int(s, 2)
		if d % 5 == 0:
			result.append(True)
		else:
			result.append(False)
	return result


def constructRectangle(area):

	result = []
	smallest = 100000
	w = np.sqrt(area)
	if isinstance(w, int):
		return [w, w]
	w = int(np.floor(w))
	while w > 0:
		if area % w == 0:
			l = area // w
			if l - w < smallest:
				smallest = l - w
				result = [l, w]
		w -= 1
	return result

def bitwiseComplement(N):

	s = bin(N)[2:]
	result = ['0' if int(i) == 1 else '1' for i in s]
	result = "".join(result)
	return int(result, 2)


def countBinarySubstrings(s):
	pass

def search(nums, target):

	mid = len(nums) // 2
	result = mid

	if target == nums[mid]:
		return result
	elif len(nums) > 1 and target > nums[mid]:
		result = result + 1 + search(nums[mid+1:], target)
	elif len(nums) > 1 and target < nums[mid]:
		result = search(nums[:mid], target)
	else:
		result = -1
	return result

def isPrime(n):

	if n <= 1:
		return False
	for i in range(2, n):
		if n % i == 0:
			return False
	return True

def countPrimes(n):

	count = 0
	for i in range(2, n):
		if i % 2 == 0:
			continue

		if isPrime(i):
			count += 1
	return count

def divisorGame(N):


	result = False
	for i in range(N-1, 0, -1):
		if N % i == 0:
			result = True
			divisorGame(N-i)
	return result

def imageSmoother(M):

	row, column = np.shape(M)
	result = np.zeros_like(M)
	print(result)

	for i in range(row):
		for j in range(column):
			count = 0
			dim = 0
			for u in range(i-1, i+2):
				for v in range(j-1, j+2):
					if 0 <= u < row and 0 <= v < column:
						dim += 1
						count += M[u][v]
			result[i][j] = int(np.floor(count/dim))
	return result

def lastStoneWeight(stones):

	i = 0
	while len(stones) > 1:
		stones = sorted(stones, reverse=True)
		if stones[i] == stones[i+1]:
			stones = stones[i+2:]
		elif stones[i] > stones[i+1]:
			stones[0] = stones[i] - stones[i+1]
			stones.remove(stones[i+1])
	if stones == []:
		return 0
	return stones[0]

def reorderLogFiles(logs):
	for log in logs:
		l = log.split()[1:]
		print(l)

def isRectangleOverlap(rec1, rec2):
	pass


def robotSim(commands, obstacles):

	from enum import Enum
	class directions(Enum):
		north = 0
		south = 1
		east = 2
		west = 3

	path = [0, 0]
	east = dict({directions.north:directions.east, directions.east:directions.south, directions.south:directions.west, directions.west:directions.north})
	west = dict({directions.north:directions.west, directions.west:directions.south, directions.south:directions.east, directions.east:directions.north})

	flag = False
	step = directions.north
	for command in commands:
		if not flag and 1 <= command <= 9:
			step = directions.north
		elif command == -1:
			step = east[step]
			flag = True
			continue
		elif command == -2:
			step = west[step]
			flag = True
			continue

		if step == directions.north:
			path[1] += command
		elif step == directions.south:
			path[1] += -command
		elif step == directions.east:
			path[0] += command
		elif step == directions.west:
			path[0] += -command

		flag = False
		print(path)


def sumEvenAfterQueries(A, queries):

	even_sum = sum(list(filter(lambda x: x % 2 == 0, A)))
	result = np.zeros_like(A)
	i = 0
	for query in queries:
		tmp = A[query[1]] + query[0]
		if tmp % 2 == 0:
			A[query[1]] = tmp
			even_sum += A[query[1]]
		else:
			even_sum -= A[query[1]]
			A[query[1]] = tmp
		result[i] = even_sum
		i += 1
	return result



# path += (0, command)
#if A[0] < 0 and K % 2 == 1:
#return -A[0] + sum(A[1:])
# if A[0] < 0 and K % 2 == 0:
# distance = [7,10,1,12,11,14,5,0]
# start = 7
# destination = 2
#
# distance =[3,6,7,2,9,10,7,16,11]
# start = 6
# destination = 2
# distance = [1,2,3,4]
# start = 0
# destination = 3
# result = distanceBetweenBusStops(distance, start, destination)
# print(result)
# nums = [0,4,3,0,4]
# nums = [3,9,7,8,3,8,6,6]
# nums = [3,6,7,7,0]
# nums = [0,0]
# nums = [1,1,2]
# # nums = [1,0,0,6,4,9]
# result = specialArray(nums)
# print(result)
# A = [4,2,3]
# K = 1
# A = [3,-1,0,2]
# K = 3
# A = [2,-3,-1,5,-4]
# K = 2
# A = [4,2,3]
# K = 1
# if abs(A[0]) < abs(A[1]):
# 	return -A[0] + sum(A[1:])
# elif abs(A[0]) > abs(A[1]):
# 	return -A[0] + -A[1] + sum(A[2:])

# A =[-2,5,0,2,-2]
# K = 3
# result = largestSumAfterKNegations(A, K)
# print(result)

# N = 2
# trust = [[1,2]]
# # N = 3
# trust = [[1,3],[2,3]]
# N = 3
# trust = [[1,3],[2,3],[3,1]]
# N = 3
# trust = [[1,2],[2,3]]
# N = 4
# trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
# result = findJudge(N, trust)
# print(result)

# nums = [0,1,2,4,5,7]
# nums = [0,2,3,4,6,8,9]
# result = summaryRanges(nums)
# print(result)


# nums = [3, 5, 9]
# nums = [1,2]
# nums = [5,6,8,8,5]
# result = minMoves(nums)
# print(result)

# g = [1,2,3]
# s = [1,1]
# result = findContentChildren(g, s)
# print(result)
# A = [0,1,1]
# result = prefixesDivBy5(A)
# print(result)

# area = 122122
# area = 4
# area = 9999999
# result = constructRectangle(area)
# print(result)

# N = 5
# result = bitwiseComplement(N)
# print(result)

# target = 1000
# nums = [-1, 0, 3, 5, 9, 12]
#
# # nums = [-1,0,3,5,9,12]
# # target = 1000
#
# result = search(nums, target)
# print(result)

# result = countPrimes(499979)
# print(result)


# N = 7
# result = divisorGame(N)
# print(result)

# M = [[1,1,1],
#  [1,0,1],
#  [1,1,1]]
# M = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
# result = imageSmoother(M)
# print(result)

# stones = [2,7,4,1,8,1]
# result = lastStoneWeight(stones)
# print(result)

# logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
# result = reorderLogFiles(logs)
# print(result)

# commands = [4,-1,4,-2,4]
# obstacles = [[2,4]]
# # commands = [4,-1,3]
# obstacles = []
# result = robotSim(commands, obstacles)
# print(result)

# A = [1,2,3,4]
# queries = [[1,0],[-3,1],[-4,0],[2,3]]
# result = sumEvenAfterQueries(A, queries)
# print(result)

class TreeNode:
	def __init__(self, val=0, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right

def bfs(l):

	if l is []:
		return None

	q = deque()
	root = TreeNode(l[0])
	q.append(root)
	i = 1
	while q.__len__() > 0:
		parent = q.popleft()
		j = 0
		while i + j < len(l) and j < 2:
			if l[i+j] is not None:
				child = TreeNode(l[i+j])
				if (i + j) % 2 == 1:
					parent.left = child
				else:
					parent.right = child
				q.append(child)
			j += 1
		i = i + j
	return root

def traverse_tree(root):

	result = []
	if root is None:
		return ""

	q = deque()
	q.append(root)
	while q.__len__() > 0:
		t = q.popleft()
		result.append(t.val)
		if t.left is not None:
			q.append(t.left)
		if t.right is not None:
			q.append(t.right)
	return result

def averageOfLevels(root):

	if root.val is None:
		return 0

	q = deque()
	q.append(root)
	i = 0
	result = []
	while q.__len__() > 0:
		tmp = []
		for _ in range(2**i):
			if q.__len__() == 0:
				break
			parent = q.popleft()
			if parent.val is not None:
				tmp.append(parent.val)
			if parent.left is not None:
				q.append(parent.left)
			if parent.right is not None:
				q.append(parent.right)
			i += 1
		print(tmp)
		result.append(sum(tmp)/len(tmp))
	return result

def averageOfLevels(root):

	if root.val is None:
		return 0

	q = [root]
	i = 0
	count = 0
	result = []
	while len(q) > 0:
		tmp = []
		parent = q[0]
		if count == 2**i - 1:
			[tmp.append(i.val) for i in q]
			print(tmp, count, i)
			result.append(sum(tmp)/len(tmp))
			i += 1
		if parent.left is not None:
			q.append(parent.left)
		if parent.right is not None:
			q.append(parent.right)
		q.remove(q[0])
		count += 1
	return result

def traverse_recursive(root):

	tmp = []
	result = []
	if not root.left and not root.right:
		return []

	if root.left:
		tmp = tmp + [root.val] + traverse_recursive(root.left)

	if root.right:
		tmp = tmp + [root.val] + traverse_recursive(root.right)
	return tmp

def levelOrderBottom(root):

	result = []
	tmp = []
	if not root.left or not root.right:
		return [root.val]

	if root.left:
		result += levelOrderBottom(root.left)

	if root.right:
		result += levelOrderBottom(root.right)

	# result.append(tmp)
	return result


# root = bfs([3, 9, 20, None, None, 15, 7])
root = bfs([5, 14, None, 1])
# root = bfs([5, 14, None, 1])

# result = traverse_tree(root)
# print(result)

result = traverse_recursive(root)
print(result)

# result = averageOfLevels(root)
# print(result)

# result = levelOrderBottom(root)
# print(result)
