from collections import deque
import re

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

def tree2str(t):
	result = ""
	def dfs(t):
		result = ""
		if t is None:
			return "()"
		result += str(t.val)
		if t.left is not None:
			result += "(" + dfs(t.left) + ")"
		elif t.right:
			result += "()"

		if t.right is not None:
			result += "(" + dfs(t.right) + ")"
		return result
	result = dfs(t)
	print(result)

def averageOfLevels(root):
	d = dict()
	level = 0
	def dfs(root, d, level):
		result = []
		if root is None:
			return

		if d.__contains__(level):
			d[level].append(root.val)
		else:
			d[level] = [root.val]

		if root.left is not None:
			dfs(root.left, d, level+1)
		if root.right is not None:
			dfs(root.right, d, level+1)
	dfs(root, d, level)
	result = []
	for k, v in d.items():
		result.append(sum(v)/len(v))
	return result


def levelOrderBottom(root):
	d = dict()
	level = 0
	def dfs(root, d, level):
		result = []
		if root is None:
			return
		if d.__contains__(level):
			d[level].append(root.val)
		else:
			d[level] = [root.val]

		if root.left is not None:
			dfs(root.left, d, level+1)
		if root.right is not None:
			dfs(root.right, d, level+1)
	dfs(root, d, level)
	result = []
	d = dict(sorted(d.items(), key=lambda kv: kv[0], reverse=True))
	for key in d.keys():
		result.append(d[key])
	return result

def leafSimilar(root1, root2):

	def dfs(root):
		result = ""
		if root.left is not None:
			result += dfs(root.left)

		if root.right is not None:
			result += dfs(root.right)

		if root.left is None and root.right is None:
			result += ','+str(root.val)

		return result

	result1 = dfs(root1)
	result2 = dfs(root2)
	result1 = result1.split(',')[1:]
	result2 = result2.split(',')[1:]
	print(result1, result2)
	return result1 == result2

def increasingBST(root):
	def dfs(root, result):
		if root.left is not None:
			dfs(root.left, result)
		if root.val is not None:
			result.append(root.val)
		if root.right is not None:
			dfs(root.right, result)
		return

	result = []
	dfs(root, result)
	root = TreeNode(result[0])
	parent = root
	for i in result[1:]:
		child = TreeNode(i)
		parent.right = child
		parent = child
	return root

def containsPattern(arr, m, k):

	for i in range(len(arr)-m+1):
		pattern = arr[i:i+m]
		print(pattern)
		for j in range(i, len(arr)-m+1):
			count = 0
			for l in range(j,len(arr)-m+1,m):
				print('new', arr[l:l+m])
				if pattern != arr[l:l+m]:
					break
					count += 1
				if count >= k:
					return True
	return False

def countBinarySubstrings(s):
	pass

def countGoodRectangles(rectangles):

	from collections import Counter
	result = []
	for rectangle in rectangles:
		result.append(min(rectangle[0], rectangle[1]))
		c = Counter(result)
		c = dict(sorted(c.items(), key=lambda kv: kv[0], reverse=True))
		v = list(c.values())
		print(v)
	return v[0]

def isSubtree(s, t):
	def dfs(root, t, flag):
		result = ""
		if flag or root.val == t.val:
			if not flag:
				flag = True
			result = ',' + str(root.val)

		if root.left is not None:
			result += dfs(root.left, t, flag)

		if root.right is not None:
			result += dfs(root.right, t, flag)
		return result

	result1 = dfs(s, t, False)
	result2 = dfs(t, t, False)
	result1 = result1.split(',')[1:]
	result2 = result2.split(',')[1:]
	print(result1, result2)

	for i in range(len(result1)):
		j = 0
		k = i
		while j < len(result2) and k < len(result1) and result1[k] == result2[j]:
			j += 1
			k += 1
		if j == len(result2) and k == len(result1):
			return True
	return False

def gcdOfStrings(str1, str2):

	if str1[0] != str2[0]:
		return ""

	def gcd(a, b):
		if b == 0:
			return a
		return gcd(b, a%b)
	l = gcd(len(str1), len(str2))
	pattern = str2[:l]
	strs = [str1, str2]
	for s in strs:
		i = 0
		while i < len(s):
			if s[i:i+l] != pattern:
				return ""
			i += l

	return pattern

def nextGreatestLetter(letters, target):

	letters = sorted(letters)
	ch = target
	for i in range(1, 27):
		ch = chr(ord('a')+(ord(target) - ord('a')+i) % 26)
		if ch in letters:
			return ch
	return ch

# root = bfs([1,2,3, 4])
# root1 = bfs([3,5,1,6,2,9,8,None,None,7,4])
# root2 = bfs([3,5,1,6,7,4,2,None,None,None,None,None,None,9,8])
# root1 = bfs([3,5,1,6,2,9,8,None,None,7,14])
# root2 = bfs([3,5,1,6,71,4,2,None,None,None,None,None,None,9,8])


# root = bfs([5,3,6,2,4,None,8,1,None,None,None,7,9])
# root = bfs([5,1,7])
# result1 = traverse_tree(root1)
# result2 = traverse_tree(root2)
# print(result1, result2)

# s = tree2str(root)
# print(s)
# result = averageOfLevels(root)
# result = levelOrderBottom(root)
# print(result)

# result = leafSimilar(root1, root2)
# print(result)

# s = "00110011"
# result = countBinarySubstrings(s)
# print(result)
#
# result = tree2str(root)
# print(result)

# result = traverse_tree(s)
# print(result)
# result = isSubtree(s, t)
# print(result)


# str1 = "ABCABC"
# str2 = "ABC"
# result = gcdOfStrings(str1, str2)
# print(result)


# letters = ["c", "f", "j"]
# target = "a"
# letters = ["c", "f", "j"]
# target = "c"
# letters = ["c", "f", "j"]
# target = "k"
# result = nextGreatestLetter(letters, target)
# print(result)

# arr = [2,2,2,2]
# m = 2
# k = 3

# arr = [1,2,1,2,1,3]
# m = 2
# k = 3

# arr = [1,2,1,2,1,1,1,3]
# m = 2
# k = 2
# arr = [1,2,4,4,4,4]
# arr = [1,2,1,2,1,1,1,3]
# m = 2
# k = 2

# arr = [2,2,1,2,2,1,1,1,2,1]
# m = 2
# k = 2
# arr = [1,2,4,4,4,4]
# m = 1
# k = 3

# arr = [1,2,1,2,1,1,1,3]
# m = 2
# k = 2

# arr = [1,2,1,2,1,3]
# m = 2
# k = 3
# arr = [1,2,1,2,1,1,1,3]
# m = 2
# k = 2

# arr = [1,2,3,1,2]
# m = 2
# k = 2

# arr = [2,2,2,2]
# m = 2
# k = 3

# arr = [2,2,1,2,2,1,1,1,2,1]
# m = 2
# k = 2

# result = containsPattern(arr, m, k)
# print(result)

# s = "00110011"
# s =  "10101"
# result = countBinarySubstrings(s)
# print(result)

# rectangles = [[5,8],[3,9],[5,12],[16,5]]
# rectangles = [[5,8],[3,9],[3,12]]
# result = countGoodRectangles(rectangles)
# print(result)


# root = bfs([5,3,6,2,4,None,8,1,None,None,None,7,9])
# result = increasingBST(root)
# result = traverse_tree(result)
# print(result)

# s = bfs([3, 4, 5, 1, 2])
# t = bfs([4, 1, 2])
# s = bfs([1, 1, None])
# t = bfs([1])
# s = bfs([1,2,3])
# t = bfs([1,2])
# result = isSubtree(s, t)
# print(result)
class MinStack:
	def __init__(self):
		"""
        initialize your data structure here.
        """
		self.heap = []
	def push(self, x: int) -> None:
		self.heap.append(x)
		# print(self.heap)
	def pop(self) -> None:
		self.heap = self.heap[0:-1]

	def top(self) -> int:
		return self.heap[-1]

	def getMin(self) -> int:
		return min(self.heap)

def invertTree(root: TreeNode) -> TreeNode:

	tmp = TreeNode()
	if root is not None:
		tmp = root.left
		root.left = root.right
		root.right = tmp
		del tmp
		invertTree(root.left)
		invertTree(root.right)
	return root

def diameterOfBinaryTree(root: TreeNode) -> int:
	level = 0
	if root:
		if root.left is not None:
			level = max(level+1, diameterOfBinaryTree(root.left))
		if root.right is not None:
			level = max(level+1, diameterOfBinaryTree(root.right))
	return level

def findJudge(N, trust) -> int:

	out_degree = [0]*N
	in_degree = [0]*N
	for record in trust:
		out_degree[record[1]-1] += 1
		in_degree[record[0]-1] += 1

	for i in range(len(out_degree)):
		if out_degree[i] == N-1 and in_degree[i] == 0:
			return i + 1
	return -1

def islandPerimeter(grid) -> int:
   def dfs(grid,x,y, visited,count):
        visited.add((x,y))
        for i, j in [[x+1,y],[x-1,y],[x,y-1],[x,y+1]]:
            if 0 <= i < len(grid) and 0<= j < len(grid[0]) and grid[i][j] ==1:
                if (i,j) not in visited:
                    dfs(grid,i,j,visited,count)
                else:
                    count[0] +=1
        return count

   for i,row in enumerate(grid):
	   for j, value in enumerate(row):
		   if value == 1:
			   return dfs(grid,i,j,set(),[0])[0]

# grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
# result = islandPerimeter(grid)
# print(result)

def sortedArrayToBST(nums) -> TreeNode:
	center = 0
	if 0 <= center < len(nums):
		center = len(nums) // 2
		val = nums[center]
		root = TreeNode(val)
		root.left = sortedArrayToBST(nums[:center])
		root.right = sortedArrayToBST(nums[center+1:])
		return root
	return None

def isCousins(root: TreeNode, x: int, y: int) -> bool:

	def dfs(root, val):
		level = 0
		if root is not None and root.val != val:
			level = 1 + dfs(root.left, val)

		if root is not None and root.val!= val:
			level = 1 + dfs(root.right, val)
		return level
	result = dfs(root, x)
	print(result)

def getAllElements(root1, root2):

	def dfs(root):
		result = []

		if root is None:
			return []

		if root.left is not None:
			result.extend(dfs(root.left))
		result.append(root.val)
		if root.right is not None:
			result.extend(dfs(root.right))
		return result

	result1 = dfs(root1)
	result2 = dfs(root2)
	result1.extend(result2)
	result1 = sorted(result1)
	return result1

# root = bfs([5,1,7,0,2])
# result = getAllElements(root)
# print(result)

def deepestLeavesSum(root) -> int:

	d = dict()
	level = 0
	def dfs(root, d, level):
		result = []
		if root is None:
			return
		if root.left is not None:
			dfs(root.left, d, level+1)
		if root.right is not None:
			dfs(root.right, d, level+1)
		if d.__contains__(level):
			d[level].append(root.val)
		else:
			d[level] = [root.val]

	dfs(root, d, 0)
	k = max(list(d.keys()))
	return sum(d[k])
# root = bfs([1,2,3,4,5,None,6,7,None,None,None,None,8])
# result = deepestLeavesSum(root)
# print(result)

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# list1 = [0,1,2,3,4,5]
# a = 3
# b = 4
# list2 = [1000000,1000001,1000002]
# x = 4
# y = 3
# root = bfs([1,2,3,4])
# result = isCousins(root, x, y)
# print(result)

# nums = [-10,-3,0,5,9]
# root = sortedArrayToBST(nums)
# root = bfs(nums)
# result = traverse_tree(root)
# print(result)


# ms = MinStack()
# ms.push(-2)
# ms.push(0)
# ms.push(-1)
# result = ms.getMin()
# print(result)
# result = ms.top()
# print(result)
# ms.pop()
# result = ms.getMin()
# print(result)

# ms.push(-2)
# ms.push(0)
# ms.push(-3)
# result = ms.getMin()
# print(result)
# ms.pop()
# result = ms.top()
# print(result)
# result = ms.getMin()
# print(result)

# root = bfs([4, 2, 7, 1, 3, 6, 9])
# root = invertTree(root)
# result = traverse_tree(root)
# print(result)

# root = bfs([1, 2, 3, 4, 5])
# root = bfs([[]])
# result = diameterOfBinaryTree(root)
# print(result)
# N = 3
# trust = [[1,3],[2,3],[3,1]]
# N = 2
# trust = [[1,2]]
# N = 3
# trust = [[1,3],[2,3]]
# N = 3
# trust = [[1,3],[2,3],[3,1]]
# N = 3
# trust = [[1,2],[2,3]]
# N = 4
# trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
# result = findJudge(N, trust)
# print(result)

# def bstToGst(root: TreeNode) -> TreeNode:
# 	dfs(root)
# 	# print(result)
#
# root = bfs([4,1,6,0,2,5,7,None,None,None,3,None,None,None,8])
# result = bstToGst(root)
# print(result)

def canArrange(arr, k: int) -> bool:
	freq = [0] * (len(arr)//2)
	for x in arr:
		val = ((x % k) + k) % k
		print(val)

def complexNumberMultiply(a: str, b: str) -> str:
	import numpy as np
	import re
	lr = [a, b]
	v = []
	for s in lr:
		s = s.replace('i', 'j')
		s = s.replace('+-', '-')
		v.append(complex(s))
	result = v[0] * v[1]
	result = str(result)
	result = result.replace('j','i')
	result = result.lstrip('(')
	result = result.rstrip(')')
	result = list(result[::-1])
	# for i in range(len(result)):
	# 	if result[i] == '-':
	# 		result.insert(i+1,'+')
	# 		break
	# val = result.find('+')
	# if val == -1:
	# 	result.insert(0, '+')
	#
	# if result[-1] == '+':
	# 	result.insert(0, '0')
	# return "".join(result[::-1])

# a = "1+1i"
# b = "1+1i"
# # a = "1+0i"
# # b = "1+0i"
# # a = "78+-76i"
# # b = "-86+72i"
# a = "20+22i"
# b = "-18+-10i"
#
# result = complexNumberMultiply(a, b)
# print(result)

# arr = [1,2,3,4,5,10,6,7,8,9]
# k = 5
# arr = [1,2,3,4,5,6]
# k = 10
# arr = [1,2,3,4,5,6]
# k = 7
# result = canArrange(arr, k)
# print(result)

def reverseWords(s) -> str:
	pass
# s = "   the sky is blue   "
# result = reverseWords(s)
# print(result)

def convertBST(root: TreeNode) -> TreeNode:

	def dfs(root):
		result = 0
		if root is None:
			return 0

		if root.right is not None:
			result = root.val + dfs(root.right)
			print(result)

		if root.left is not None:
			result = result + dfs(root.left)
			print(result)

		if root.left is None and root.right is None:
			return result + root.val

		return result
	dfs(root)

# root = bfs([4,1,6,0,2,5,7,None,None,None,3,None,None,None,8])
root = bfs([1,2,3,4,5])
result = convertBST(root)
print(result)