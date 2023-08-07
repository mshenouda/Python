
from bisect import bisect_left
from copy import deepcopy
from dis import dis
from functools import reduce
from heapq import heappush
from inspect import _void
import os
import sys
from collections import deque
import collections
from pip import List
from collections import Counter
from collections import defaultdict
import bisect

class TreeNode:
	def __init__(self, val=0, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right


def lengthOfLongestSubstring(s) -> int:
    longest = [] 
    sub = set()
    for a in s:
        if sub.__contains__(a):
            longest.append(len(sub))
        else:
            sub.add(a)

    k = sorted(longest)[-1]
    l = len(sub)

    if k > l:
        return k 
    return l

def maxSubsequence(nums, k):
    import heapq
    values = deepcopy(nums)
    heapq.heapify(nums)
    result = []

    for _ in range(len(nums)):
        v = heapq.heappop(nums)
        result.append(v)

    indices = []   
    result = result[-k:]
    for val in result:
        for i, value in enumerate(values):
            if val == value:
                indices.append(i)

    indices.sort()
    result = []
    result = list(map(lambda x: values[x], indices))
    return result    

def countValidWords(sentence: str) -> int:
    import re;
    words = sentence.split(' ')
    words = list(filter(lambda x: x!= '', words))
    count = 0
    # pattern = r"^[\.\,\!]$|^(?P<letters>[a-z]+)|(?P=letters)\-?(?P=letters)[\.\,\!]?$"
    pattern = r"^[\.\,\!]$|^([a-z]+)(\-?([a-z]+))?[\.\,\!]?$"
    for word in words:
        if re.search(pattern, word) and word[-1] != '-':
            count+=1
        
            print(word)
    return count
        
def restoreIpAddresses(s: str) -> list[str]:
    pass

def findCenter(edges) -> int:
    map = {}
    for edge in edges:
        u, v = edge
        def fillMap(x, y):
            if not map.__contains__(x):
                map[x] = [y]
            else:
                map[x]+=[y] 
        fillMap(u, v)
        fillMap(v, u)
    n = len(map)
    


def numSteps(s: str) -> int:
    
    def add1(l: list) -> list[int]:
        carry = 0
        n = len(l)
        i = n-1
        while i > -1:
            if i == n-1:
                carry = 1 if l[i] == '1' else 0
            if l[i] == '1':
                carry = 1 if carry == 1 else 0
                l[i] = '0'
            else:
                l[i] = '1'
                carry = 0
            if carry == 0 and i > -1:
                break
            i-=1
        
        if carry == 1:
            l = ['1'] + l 
        return l 

    def shiftRight1(l):
        l = l[0:-1]
        return l


    l = list(s)        
    result = 0       
    while len(l) > 1:
        if l[-1] == '1':
            l = add1(l) 
            result += 1
        else:
            l = shiftRight1(l)
            result += 1    
    return result

def canPartitionKSubsets(nums, k: int) -> bool:
    val = sum(nums)//k
    pass

def dijkstra(n, graph, k):
    import math
    V = n
    values = [math.inf] * V
    parent = [0] * V
    processed = [False] * V
    
    parent[k-1] = -1
    values[k-1] = 0

    def findMinimum():
        minimum = math.inf
        vertex = 0
        for i in range(V):
            if processed[i] == False and values[i] < minimum:
                vertex = i
                minimum = values[i]
        return vertex   
    
    def dfs():
        for _ in range(V):
            U = findMinimum()
            print(U)
            processed[U] = True
            for j in range(V):
                if graph[U][j] != 0 and processed[j] == False and (values[U] != math.inf) and (values[U] + graph[U][j] < values[j]):
                    values[j] = values[U] + graph[U][j]
                    parent[j] = U
    
    dfs()
    return values, parent

def OppositeDijkstra(n, graph, k):
    import math
    V = n
    values = [-math.inf] * V
    parent = [0] * V
    processed = [False] * V
    
    parent[0] = -1
    values[k-1] = 0

    def findMaximum():
        maximum = -math.inf
        vertex = 0
        for i in range(V):
            if processed[i] == False and values[i] > maximum:
                vertex = i
                maximum = values[i]
        return vertex   
    
    def dfs():
        for i in range(V-1):
            U = findMaximum()
            processed[U] = True
            for j in range(V):
                if graph[U][j] != 0 and processed[j] == False and (values[U] != math.inf) and (values[U] + graph[U][j] > values[j]):
                    values[j] = values[U] + graph[U][j]
                    parent[j] = U
    
    dfs()
    return values, parent
  
 
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
     
def sumNumbers(root) -> int: 

    def computeSum(array):
        result = 0
        for val in array:
            result = 10 * result + val
        return result
    
    def dfs(root, array=[]):
        if root == None:
           return 0
       
        sum = 0
        array += [root.val]
        if root.left == None and root.right == None: 
            sum = computeSum(array) 
        if root.left != None:
            sum += dfs(root.left, array)
        if root.right != None:
            sum += dfs(root.right, array)
        array.pop()
        return sum
    
    result = dfs(root)
    return result

def removeLeafNodes(root: TreeNode, target: int) -> TreeNode:
    pass

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

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

def levelOrder(root: Node):
    obj = {}
    def dfs(root, level=0):
        if root == None:
            return
        if obj.get(level) == None:
            obj[level] = [root.val]
        else:
            obj[level] += [root.val]
        for i in range(root.children):
            dfs(root[i], level+1)
    dfs(root)
    print(obj)    
    
def canVisitAllRooms(rooms) -> bool:
    n = len(rooms)
    visited = [False] * n
    def dfs(i):
        count = 0
        if visited[i]:
            return 0
        else:
            visited[i] = True
            count = 1          
        
        for j in rooms[i]:
            count += dfs(j)
        return count
    
    count = dfs(0)
    if count == n:
        return True
    return False
    
def numOfMinutes(n: int, headID: int, manager, informTime) -> int:
    pass
    
def goodNodes(root: TreeNode) -> int:
    import math
    def dfs(root, maxNum=-math.inf):
        count = 0
        if root.val >= maxNum:
            maxNum = root.val
            count = 1
        if root and root.left != None:
            count += dfs(root.left, maxNum)
        if root and root.right != None:
            count += dfs(root.right, maxNum)
        return count
        
    count = dfs(root)
    return count
    
def longestUnivaluePath(root) -> int:    
    import math
    global maxNum
    global count
    maxNum = 0
    count = 0
    def dfs(root, value = -math.inf):
        global maxNum
        global count
        if root == None:
            return
        
        if value == root.val:
            count += 1
        else:
            count = 0
        if root != None and root.left != None:
            dfs(root.left, root.val)
        if root != None and root.right != None:
            dfs(root.right, root.val)
        maxNum = max(maxNum, count)
        return
    
    count = dfs(root)
    return maxNum
   
def maximumDetonation(bombs: List[List[int]]) -> int:   
    n = len(bombs)
    adjList = [[] for _ in range(n)]
    for i,bomb in enumerate(bombs):
        x1, y1, r = bomb
        for j, bomb in enumerate(bombs):
            if i != j:
                x2, y2, _ = bomb
                dist = (x1-x2)**2 + (y1-y2)**2
                radius = r**2
                #print(dist, radius) 
                if  dist <= radius: 
                    adjList[i] += [j]
    
        
    def dfs(i, visited):
        visited[i] = True
        result = 1
        for j in adjList[i]:
            if not visited[j]:    
                result += dfs(j, visited)
        return result
        
    maxNum = 0
    for i in range(n):
        visited = [False]*n
        result = dfs(i, visited)
        maxNum = max(maxNum, result)
    return maxNum
    
def findCircleNum(isConnected) -> int:
    n = len(isConnected) 
    visited = [False] * n
    def dfs(i):
        visited[i] = True
        for j, status in enumerate(isConnected[i]):
            if not visited[j] and status == 1:
                    dfs(j)
                    
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            dfs(i)
    return count

def minimumOperations(nums: List[int], start: int, goal: int) -> int:
    import math
    n = len(nums)
    dp = {}
    def dfs(x):
        if x > 1000 or x < 0: return 0
        if x == goal: return 0
        if dp.get(x): return dp[x]
        counter = 1
        result = math.inf
        for num in nums:
            counter += dfs(x+num) + dfs(x^num)
            #counter += dfs(x-num)
            result = min(result, counter) 
            dp[x] = result
        return result     
    
    result = dfs(start)
    print(dp)
    return result
    
def printTree(root) -> List[List[str]]:    
    def dfs(root):
        if root == None:
            return 0
        height = 0
        if root.left != None:
            height =  max(height, 1 + dfs(root.left))
        if root.right != None:
            height =  max(height, 1 + dfs(root.right))
        return height
    
    height = dfs(root)
    n = 2**(height+1) - 1
    print(height, n)    
        
    def bfs(root):
        import math
        result = [[""]*n for _ in range(height+1)]
        queue = []
        r = 0
        c = (n-1)//2
        queue.append((root, r, c))
        
        while len(queue) > 0:
            node = queue.pop(0)
            new_node, r, c = node[0], node[1], node[2] 
            result[r][c] = str(new_node.val)
            if new_node.left:
                queue.append((new_node.left, r+1, c-2**(height-r-1)))
            
            if new_node.right:
                queue.append((new_node.right, r+1, c+2**(height-r-1)))      
    
        return result    
    result = bfs(root)
   
def averageOfSubtree(root) -> int:
    import math
    dp = {}
    # i = 0
    global sum
    global count
    sum = 0
    #count = 0
    global count
    count = []  
       
    def dfs(root):
        # global sum, count
        global count
        
        if root == None:
            return 0
        
        total = 1
        count += [total]
        if root.left:
            total += dfs(root.left)
        if root.right:
            total += dfs(root.right)

        # sum += root.val
        # count += 1
        # print(sum, count)
        # if  math.floor(sum/count) == root.val:
        #     total = 1    
        # return total
    
    result = dfs(root)
    print(count)
    return result
   
def bstToGst(root: TreeNode) -> TreeNode:
    def dfs(root):
        if root == None:
            return 0  
        total = 0 
        if root.right:
            total += dfs(root.right)
        total = root.val
        if root.left:
            total += dfs(root.left)
        return total
    dfs(root)
  
def isValidSerialization(preorder: str) -> bool:
    def dfs(root) -> list:
        if root:
            val = [str(root.val)]
        if root.left:
            val += dfs(root.left)
        else:
            val += ["#"]
             
        if root.right:
            val += dfs(root.right)
        else:
            val += ["#"]
        
        return val 

def findCheapestPrice(n: int, flights, src: int, dst: int, k: int) -> int:
    
    matrix = [[0]*n for _ in range(n)]
    for edge in flights:
        i, j, cost = edge
        matrix[i][j] = cost
    
    print(matrix)
    dist, parent = dijkstra(n, matrix)
    print(dist, parent)

def countRestrictedPaths(n, edges) -> int:
    matrix = [[0]*n for _ in range(n)]
    for edge in edges:
        i, j, cost = edge
        matrix[i-1][j-1] = cost
    
    print(matrix)    
    dist, parent = dijkstra(n, matrix, n)
    print(dist, parent)
   
if __name__ == '__main__':
    pass

    # n = 5
    # edges = [[1,2,3],[1,3,3],[2,3,1],[1,4,2],[5,2,2],[3,5,1],[5,4,10]]
    # result = countRestrictedPaths(n, edges)
    # print("Result is {0}".format(result))
 
    # n = 4
    # flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
    # src = 0
    # dst = 3
    # k = 1
    # result = findCheapestPrice(n, flights, src, dst, k)
    # print("Result is {0}".format(result))
    
    # times = [[2,1,1],[2,3,1],[3,4,1]]
    # n = 4
    # k = 2
    # times = [[3,5,78],[2,1,1],[1,3,0],[4,3,59],[5,3,85],[5,2,22],[2,4,23],[1,4,43],[4,5,75],[5,1,15],[1,5,91],[4,1,16],[3,2,98],[3,4,22],[5,4,31],[1,2,0],[2,5,4],[4,2,51],[3,1,36],[2,3,59]]
    # n = 5
    # k = 5    
    # result = networkDelayTime(times, n, k)
    # print("Result is {0}".format(result))
   
         
    
    # numCourses = 4
    # prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    # numCourses = 2
    # prerequisites = [[0,1],[1,0]]
    # result = findOrder(numCourses, prerequisites)
    # print("Result is {0}".format(result))
    
    
    # prerequisites = [[1,0],[0,1]]
    # result = canFinish(2, prerequisites)
    # print("Result is {0}".format(result))
    
    
    
    #nums = [9,3,2,4,1,None,6]
    # nums = [4,8,5,0,1,None,6]
    # root = bfs(nums)
    # # preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#" 
    # result = isValidSerialization(preorder)
    # print("Result is {0}".format(result))
    
    # result = bstToGst(root)
    # result = traverse_tree(root)
    # print("Result is {0}".format(result))
    
    # result = averageOfSubtree(root)
    # print("Result is {0}".format(result))
    
     
    #nums = [1,2]
    # nums = [1,2,3,None,4]
    # root = bfs(nums)
    # result = printTree(root)
    # print("Result is {0}".format(result))
    
    # nums = [2,4,12]
    # start = 14
    # goal = 12
    # result = minimumOperations(nums, start, goal)
    # print("Result is {0}".format(result))
        
    #isConnected = [[1,1,0],[1,1,0],[0,0,1]]
    # isConnected = [[1,0,0],[0,1,0],[0,0,1]]
    # result = findCircleNum(isConnected)
    # print("Result is {0}".format(result))
    
    
    # bombs = [[1,1,5],[10,10,5]]
    # bombs = [[2,1,3],[6,1,4]]
    # bombs = [[1,2,3],[2,3,1],[3,4,2],[4,5,3],[5,6,4]]
    # result = maximumDetonation(bombs)
    # print("Result is {0}".format(result))
    
    # nums = [5,4,5,1,1,5]
    # nums = [1,4,5,4,4,5]
    # root = bfs(nums)
    # result = longestUnivaluePath(root)
    # print("Result is {0}".format(result))
    
    
    
    #nums = [3,1,4,3,None,1,5]
    # nums = [3,3,None,4,2]
    # root = bfs(nums)
    # result = goodNodes(root)
    # print("Result is {0}".format(result))
    
    
    # n = 1
    # headID = 0
    # manager = [-1]
    # informTime = [0]
    # result = numOfMinutes(n, headID, manager, informTime)
    # print("Result is {0}".format(result))
    
    #rooms = [[1],[2],[3],[]]
    # rooms = [[1,3],[3,0,1],[2],[0]]
    # result = canVisitAllRooms(rooms)
    # print("Result is {0}".format(result))
    
    # nums = [1,2,3]
    # nums = [4,9,0,5,1]
    # root = bfs(nums)
    # result = sumNumbers(root)
    # print("Result is {0}".format(result))
    
    # nums = [1,None,3,2,4,None,5,6]
    # root = bfs(nums)
    # result = levelOrder(root)
    # print("Result is {0}".format(result))
   
    
    # n = 6
    # #edges = [[0,1], [0, 2], [1, 3], [2, 3]]
    # edges = [[5,0], [5, 3], [3, 0], [4, 0], [3,2], [2,1],[4,1]] 
    # #edges = [[0,1],[1,2],[2,3],[3,1]]
    # result = spanningTree(n, edges)
    # print("Result is {0}".format(result))
    
    # str = 'bbbbb'
    # str = 'abcabcbb'
    # str = 'pwwkew'
    # str = 'aab'
    # str = 'dvdf' 
    # s = Solution.lengthOfLongestSubstring(str)
    # nums = [2,1,3,3]
    # k = 2
    # nums = [-1,-2,3,4]
    # k = 3
    # nums = [50,-75]
    # k = 2
    # nums s= [2,1,3,3]
    # k = 2
    # result = Solution.maxSubsequence(nums, k)
    # print(f"Result is {result}")
    # sentence = "cat and  dog"
    # result = Solution.countValidWords(sentence)
    # print(result)
    # str = "101023"
    # result = Solution.restoreIpAddresses(str)
    # print("Result is  {0}".format(result))
    # sentence = "cat and  dog"
    # sentence = "alice and  bob are playing stone-game10"
    # sentence ="he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."
    # sentence =". ! 7hk  al6 l! aon49esj35la k3 7u2tkh  7i9y5  !jyylhppd et v- h!ogsouv 5"
    # sentence = " 62   nvtk0wr4f  8 qt3r! w1ph 1l ,e0d 0n 2v 7c.  n06huu2n9 s9   ui4 nsr!d7olr  q-, vqdo!btpmtmui.bb83lf g .!v9-lg 2fyoykex uy5a 8v whvu8 .y sc5 -0n4 zo pfgju 5u 4 3x,3!wl  fv4   s  aig cf j1 a i  8m5o1  !u n!.1tz87d3 .9    n a3  .xb1p9f  b1i a j8s2 cugf l494cx1! hisceovf3 8d93 sg 4r.f1z9w   4- cb r97jo hln3s h2 o .  8dx08as7l!mcmc isa49afk i1 fk,s e !1 ln rt2vhu 4ks4zq c w  o- 6  5!.n8ten0 6mk 2k2y3e335,yj  h p3 5 -0  5g1c  tr49, ,qp9 -v p  7p4v110926wwr h x wklq u zo 16. !8  u63n0c l3 yckifu 1cgz t.i   lh w xa l,jt   hpi ng-gvtk8 9 j u9qfcd!2  kyu42v dmv.cst6i5fo rxhw4wvp2 1 okc8!  z aribcam0  cp-zp,!e x  agj-gb3 !om3934 k vnuo056h g7 t-6j! 8w8fncebuj-lq    inzqhw v39,  f e 9. 50 , ru3r  mbuab  6  wz dw79.av2xp . gbmy gc s6pi pra4fo9fwq k   j-ppy -3vpf   o k4hy3 -!..5s ,2 k5 j p38dtd   !i   b!fgj,nx qgif "
    # sentence = "!this  1-s b8d!"
    # result = Solution.countValidWords(sentence)
    # print("Result is  {0}".format(result))
    
    # nums = [10,1,2,4,7,2]
    # limit = 5
    # result = Solution.longestSubarray(nums, limit)
    # print("Result is  {0}".format(result))
    
    # edges = [[1,2],[2,3],[4,2]]
    # result = Solution.findCenter(edges);
    # print("Result is  {0}".format(result))
        
    #graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
    # graph = [[1,3],[0,2],[1,3],[0,2]]
    #graph = [[],[2,4,6],[1,4,8,9],[7,8],[1,2,8,9],[6,9],[1,5,7,8,9],[3,6,9],[2,3,4,6,9],[2,4,5,6,7,8]]
    # result = Solution.isBipartite(graph)
    # print("Result is {0}".format(result))
    
    # n = 8 
    # edgeList = [[0,3],[0,4],[1,3],[2,4],[2,7],[3,5],[3,6],[3,7],[4,6]]
    # result = Solution.getAncestors(n, edgeList)
    # print("Result is {0}".format(result))
    
    #s = "1011"
    # s = "10"
    # result = Solution.numSteps(s)
    # print("Result is {0}".format(result))
    
    # nums = [4,3,2,3,5,2,1]
    # k = 4
    
    # nums = [1,2,3,4]
    # k = 3   
    # result = Solution.canPartitionKSubsets(nums, k)
    # print("Result is {0}".format(result))
   
#    n = 5
#    edges = [[0,3], [2, 4]]
#    result = Solution.adjMatrix(n, edges)
#    print("Result is {0}".format(result))
   
#    n = 5
#    edges = [[1,2], [1, 3], [1, 4], [2, 4]]
#    result = Solution.adjList(n, edges)
#    print("Result is {0}".format(result))
     
#    n = 5
#    edges = [[1,2], [1, 3], [3, 1]]
#    result = traverseDfs(n, edges)
#    print("Result is {0}".format(result))
   
#    n = 5
#    edges = [[1,2], [1, 3], [1, 4], [4, 2]]
#    result = traverseBfs(n, edges)
#    print("Result is {0}".format(result))
      
    # n = 4
    # edges = [[0,1], [1, 2], [2, 3], [3, 1]]
    # result = detectCylce(n, edges)
    # print("Result is {0}".format(result))

    # n = 4
    # edges = [[0,1], [1, 2], [2, 3]]
    # result = detectCycleUnDirected(n, edges)
    # print("Result is {0}".format(result))
   
    #n = 6
    #edges = [[0,1], [0, 2], [1, 3], [2, 3]]
    #edges = [[5,0], [5, 3], [3, 0], [4, 0], [3,2], [2,1],[4,1]] 
    #edges = [[0,1],[1,2],[2,3],[3,1]]
    # result = topologicalSortKahn(n, edges)
    # print("Result is {0}".format(result))
    
    # graph = [[0, 1, 4, 0, 0, 0],
    #     [1, 0, 4, 2, 7, 0],
    #     [4, 4, 0, 3, 5, 0],
    #     [0, 2, 3, 0, 4, 6],
    #     [0, 7, 5, 4, 0, 7],
    #     [0, 0, 0, 6, 7, 0]]
    
    #dijkstra(graph)
    
    # points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    # minCostConnectPoints(points)
    
def pathSum(root, target):
    ans = 0
    cache = collections.defaultdict(int);
    cache[0] = 1
    
    def dfs(root, cur_sum):
        if not root:
            return
        cur_sum += root.val
        ans += cache[cur_sum - target] 
        # like two sum.
        # If complement of current sum is seen before then then there exists those many target paths to this node
        cache[cur_sum] += 1 # keep on adding cur_sum at each node to the cache
        dfs(root.left, cur_sum)
        dfs(root.right, cur_sum)
        cache[cur_sum] -= 1 # if a node is in current path only then its cache value > 0
       
    dfs(root, 0)
    return ans

# nums = [10,5,-3,3,2,None,11,3,-2,None,1]
# root = bfs(nums)
# targetSum = 8
# result = pathSum(root, targetSum)
# result = traverse_tree(root)
#print(result)
def wordBreak(s: str, wordDict) -> bool:
    n = len(s)
    dp = [False]*(n+1)
    dp[n] = True
    for i in range(n-1, -1, -1):
        for w in wordDict:
            if i+len(w) <= len(s) and s[i:i+len(w)] == w:
                dp[i] = dp[i+len(w)]
            if dp[i]:
                break
    print(dp)
    return dp[0]
# s = "leetcode"
# wordDict = ["leet","code"]
# s = "cars"
# wordDict = ["car","ca","rs"]
# s = "aaaaaaa"
# wordDict = ["aaaa","aaa"]
# s = "aaaaaaa"
# wordDict = ["aaaa","aaa"]
# result = wordBreak(s, wordDict)
# print('The result is {}'.format(result))
def isAdditiveNumber(num: str) -> bool:
    n = len(num)
    if n < 3: return False
    def valid(s):
        if s == "0": return True
        if s[0] == "0": return False
        return True
    
    def additive(indx, temp):
        if indx == n and len(temp) >= 3:
            return True
        for i in range(indx, len(num)):
            _s = num[indx: i+1]
            if valid(_s):        
                if len(temp) < 2 or temp[-1] + temp[-2] == int(_s):
                    temp.append(int(_s))
                    if additive(i+1, temp):
                        return True
                    temp.pop()
        return False
    
    return additive(0, [])
    
#num="112358"
# num="199100199"
# result = isAdditiveNumber(num)
# print('The result is {}'.format(result))
def carPooling(trips: List[List[int]], capacity: int) -> bool:
    trips.sort(key=lambda x: (x[1], x[2]))
    d = defaultdict(int)
    for trip in trips:
        c, f ,t = trip
        d[f] += c
        d[t] -= c

    print(d)
    res = 0
    d = sorted(d.items(), key=lambda kv: (kv[0], kv[1]))
    print(d)
    # for k, v in d.items():
    #     res += v
    # if res > capacity: return False
    # return True 
    
    
# trips = [[3,3,7], [2,1,5]]
# capacity = 4
# trips = [[3,2,8],[4,4,6],[10,8,9]]
# capacity = 11
# trips = [[3,2,7],[3,7,9],[8,3,9]]
# capacity = 11
# result = carPooling(trips, capacity)
# print('The result is {}'.format(result))

class RLEIterator:
    
    def __init__(self, encoding):
        self.q = deque([])
        l = []
        for i in range(0, len(encoding), 2):
            k = encoding[i]
            v = encoding[i+1]
            for _ in range(k):
                self.q.append(v) 
        
    def next(self, n: int) -> int:
        result = 0
        for _ in range(n):
            if self.q:
                result = self.q.popleft()
        return result if self.q else -1
    
# rLEIterator = RLEIterator([3, 8, 0, 9, 2, 5])
# result = rLEIterator.next(2)
# print(result)

# result = rLEIterator.next(1)
# print(result)

# result = rLEIterator.next(1)
# print(result)

# result = rLEIterator.next(2)
# print(result)

def removeKdigits(num: str, k: int) -> str:
    stack = []
    for c in num:
        while k > 0 and stack and stack[-1] > c:
            k-=1
            stack.pop()
        stack.append(c)
    stack = stack[:len(stack)-k]
    res = "".join(stack)
    return str(int(res)) if res else -1 
           
# num = "1432219"
# k = 3
# result = removeKdigits(num, k)
# print('The result is {}'.format(result))

def findRadius(houses, heaters) -> int:
    heaters = [float("-inf")] + heaters + [float("inf")]
    
    def bs(house):
        l, r = heaters[0], heaters[-1]
        least = 0
        while l < r:
            mid = l + (r-l) // 2
            if house <= mid:
                least = mid
                mid = r - 1
            else:
                mid = l + 1
        return l

    for house in houses:
        result = bs(house)
        maxVal = 0
        minVal = float("inf")
        radius = min(minVal, max(maxVal, result))
    return radius
        
    
# houses = [1,2,3,4]
# heaters = [1,4]
# houses = [1,2,3]
# heaters = [2]
# result = findRadius(houses, heaters)
# print('The result is {}'.format(result))

class TweetCounts:
    
    def __init__(self):
        self.d = defaultdict(list)
        self.freq = {"minute": 60, "hour": 3600, "day": 86400}
    
    def recordTweet(self, tweetName: str, time: int) -> None:
        self.d[tweetName] += [time]

    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        s = sorted(self.d[tweetName])
        result = []      
        for time in range(startTime, endTime+1, self.freq[freq]):
            l = bisect.bisect_left(s, time)
            r = bisect.bisect_right(s, min(time+self.freq[freq]-1, endTime))
            result.append(r-l)
        return result
            
# tweetCounts = TweetCounts()
# tweetCounts.recordTweet("tweet3", 0)
# tweetCounts.recordTweet("tweet3", 60)
# tweetCounts.recordTweet("tweet3", 10)
# result = tweetCounts.getTweetCountsPerFrequency("minute", "tweet3", 0, 59)
# print('The result is {}'.format(result))
# result = tweetCounts.getTweetCountsPerFrequency("minute", "tweet3", 0, 60)
# print('The result is {}'.format(result))
# tweetCounts.recordTweet("tweet3", 120)
# result = tweetCounts.getTweetCountsPerFrequency("hour", "tweet3", 0, 210)
# print('The result is {}'.format(result))
