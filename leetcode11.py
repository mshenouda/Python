<<<<<<< HEAD
from operator import indexOf

from numpy import Inf

def minSwaps(s: str) -> int:
    opened = 0
    closed = 0
    n = len(s) 
    last = n-1
    s = [letter for letter in s]
   
    def swap(last):
        for j in range(last, -1, -1):
            if s[j] == '[':
                s[j] = ']'
                last = j-1
                return
    
    counter = 0
    for i, letter in enumerate(s):
        opened += 1 if (letter == '[') else 0
        closed += 1 if (letter == ']') else 0  
        if closed > opened:
            s[i] = '['
            swap(last)
            opened += 1
            closed -= 1
            counter += 1
    return counter
      
def minRemoveToMakeValid(s: str) -> str:
    s = [letter for letter in s]
    o = 0
    c = 0
    n = len(s)
    # forward
    for i, letter in enumerate(s):
        if letter == '(':
            o += 1
        if letter == ')':
            c += 1
            if c > o:
                s[i] = ""
                c -= 1
    
    o = 0
    c = 0
    # backward
    for i in range(n-1, -1 ,-1):
        if s[i] == ')':
            c += 1
        if s[i] == '(':
            o += 1
            if o > c:
                s[i] = ""
                o -= 1
    
    return "".join(s)


def hIndex(citations) -> int:
    # import heapq
    # h = []
    # for c in citations:
    #     heapq.heappush(h, -c)
    
    # ans = 0

    # while h:
    #     x = heapq.heappop(h)
    #     x = -x
    #     if x > ans:
    #         ans += 1
    
    # return ans
    dic = {}
    cit_sorted = sorted(citations)
    for i in range(len(cit_sorted)):
        if cit_sorted[i] not in dic:
            dic[cit_sorted[i]] = len(cit_sorted[i:])
        h_index = 0
    
    print(dic)
    for key in dic:
        if dic[key] >= h_index:
            if dic[key] < key:
                h_index = dic[key]
            else:
                h_index = key
    return h_index

def wiggleSort(nums) -> None:
    nums = sorted(nums)
    n = len(nums)
    mid = n // 2 if n % 2 == 0 else n // 2 + 1 
    
    lower = [0]*mid
    upper = [0]*(n - mid) 
    
    j = 0
    for i, value in enumerate(nums):
        if i < mid:
            lower[i] = value
        else:
            upper[j] = value
            j += 1
    
    nums = [0] * n
    for i in range(len(lower)):
        nums[2*i] = lower[i]
        if (2*i + 1) < n:
            nums[2*i + 1] = upper[i] 
    
    return nums

def canPartitionKSubsets(nums, k: int) -> bool:
    import functools
    sum = functools.reduce(lambda x, y: x + y, nums)
    target = sum//k
    n = len(nums)
    print(n, target, nums)
    visited = [False] * n
    dp = {}

    global count
    count = 0
    
    def dfs(j, sum):
        visited[j] = True
        if sum == 0: return 1
        if dp.get(sum): return dp[sum]
        
        count = 0
        for i in range(j+1, n):
            if sum - nums[i] >= 0:
                count += dfs(i, sum-nums[i])
                dp[sum] = count
        return count     
    
    result = 0
    for i in range(n):
        if not visited[i]:    
            if nums[i] == target:
                # if dp.get(nums[i]):
                #     dp[nums[i]] += 1
                result += 1
            else:
                result += dfs(i, target-nums[i])
                                    
    print(n, k, dp, result, visited)
       

def wordBreak(s: str, wordDict) -> bool:
    dp = {}
    for word in wordDict:
        dp[word] = False

    def dfs(s):
        if s == "": return True

        result = False    
        for word in wordDict:
            index = s.find(word)
            if not result and index == 0:
                if dp[word]: return True
                dp[word] = True
                news = s.removeprefix(word)
                if news == "": return True
                result = True and dfs(news)
        return result
    result = dfs(s)
    print(dp, result)
    return result
    
class RLEIterator:
    def __init__(self, encoding):
        self.encoding = []
        n = len(encoding)
        i = 0
        while i < n:
            self.encoding += [encoding[i+1]]*encoding[i]
            i += 2
        print(self.encoding) 

    def next(self, n: int) -> int:
        if len(self.encoding) < n:
            return -1 
        for _ in range(n):
            last = self.encoding.pop(0)
        return last
        
#s = "][]["
# s = "]]][[["
# result =  minSwaps(s)
# print("Result is {0}".format(result)) 

#s = "lee(t(c)o)de)"
#s = "))(("
#s = "(a(b(c)d)"
# result = minRemoveToMakeValid(s)
# print("Result is {0}".format(result)) 
# citations = [3,0,6,1,5]
# result = hIndex(citations)
# print("Result is {0}".format(result)) 

# nums = [1,5,1,1,6,4]
#nums = [1,2,1,3,2]
# result = wiggleSort(nums)
# print("Result is {0}".format(result)) 

# rLEIterator = RLEIterator([3, 8, 0, 9, 2, 5])
# result = rLEIterator.next(2)
# print(result) 
# result = rLEIterator.next(1)
# print(result)
# result = rLEIterator.next(1)
# print(result)
# result = rLEIterator.next(2)
# print(result)

# nums = [4,3,2,3,5,2,1]
# k = 4

# nums = [1,2,3,4]
# k = 3
# nums = [1,1,1,1,2,2,2,2]
# k = 4
# nums = [1,2,3,5]
# k = 2
# nums = [4,5,9,3,10,2,10,7,10,8,5,9,4,6,4,9]
# k = 5
# result = canPartitionKSubsets(nums, k)
# print("Result is {0}".format(result)) 


# s = "leetcode"
# wordDict = ["leet","code"]
# s = "applepenapple"
# wordDict = ["apple","pen"]
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# s = "cars"
# wordDict = ["car","ca","rs"]
# wordDict = ["ca","rs", "car"]
# s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
# wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
# s = "aaaaaaa"
# wordDict = ["aaaa","aa"]
# result = wordBreak(s, wordDict)
# print("Result is {0}".format(result)) 

def dailyTemperatures(temperatures):
    from collections import deque
    n = len(temperatures)
    result = [0]*n
    stack = []
    l = -1
    for i, currTmp in enumerate(temperatures):           
        while len(stack) > 0 and stack[l][1] < currTmp:
            index, _ = stack.pop()
            result[index] = i - index
            l -= 1
        stack.append((i, currTmp))
        l += 1
    return result     
        
# temperatures = [73,74,75,71,69,72,76,73]
# #temperatures = [30,60,90]
# result = dailyTemperatures(temperatures)
# print("Result is {0}".format(result)) 

def generateParenthesis(n: int):
    def dfs(n):
        if n == 0: return ""
        s = ""
        s += dfs(n-1)
        s += '(' +')'
        return s
    result = dfs(n)
    return result
# n = 4
# result = generateParenthesis(n)
# print("Result is {0}".format(result)) 

def numSubseq(nums, target) -> int:     
    nums = sorted(nums)
    counter = 0
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if j >= i and a + b <= target:
                counter += 1
                counter  = counter % ((10**9)+7) 
    return counter
# nums = [3,5,6,7]
# target = 9
# result = numSubseq(nums, target)
# print("Result is {0}".format(result)) 

def myPow(x: float, n: int) -> float:
    import math
    dp = {}
    def dfs(n):
        if n == 0: return 1
        if dp.get(n): return dp[n]
        if n < 0:
            dp[n] = 1/x * dfs(n+1)
        else:
            dp[n] = x * dfs(n-1)
        return dp[n]
    
    result = dfs(n)
    print(dp)                                     
    return result
    
    
# x = 2.00000
# n = 10
# x = 2.10000
# n = 3
# x = 2.00000
# n = -2
# result = myPow(x, n)
# print("Result is {0}".format(result)) 

def removeKdigits(num: str, k: int) -> str:
    n = len(num)
    visited = [False] * n

    # def convert2Num(s: str):

    def dfs(i, count=0):
        if count < k:
            visited[i] = True
        
        
        for j, _ in enumerate(num, start=i):
            if count < k-1:
                dfs(j+1, count+1)    
            
            tmp = ""
            for v, value in enumerate(visited):
                if not value: 
                    tmp += num[v]
            print(tmp)                  
            visited[i] = False
            return
    dfs(0)       
# num = "1432219"
# k = 3
# result =removeKdigits(num, k) 
# print("Result is {0}".format(result)) 

def findLeastNumOfUniqueInts(arr, k: int) -> int:
    n = len(arr)
    visited = [False] * n
    import math

    def returnLen():
        s = set()
        for v in range(n):
            if not visited[v]:
                s.add(arr[v])
        return len(s)
    
    def dfs(i, k):
        if k < 0: return 0  
        l = 0
        result = math.inf
        for j in range(i, n):
            visited[j] = True
            result = min(result, dfs(j+1, k-1))        
            visited[j] = False
            l = returnLen()
            if k == 0:
                result = l
        return result
    
    try:       
        result = dfs(0, k)
    except UnboundLocalError as e:
        print(e)
    return result

#arr = [1,2,3]        
#arr = [4,3,1,1]
# arr = [4,3,1,1,3,3,2]
# k = 3
#k = 2
#k = 2
# result =findLeastNumOfUniqueInts(arr, k) 
# print("Result is {0}".format(result)) 

def isAdditiveNumber(num: str) -> bool:
    
    if len(num) < 3: return False
    #if num[0] == '0': return False
    
    n = len(num)
    def dfs(counter=0, start=0, lastVal = 0, a= None, b=None, s="") -> bool:
        print("start {0}, lastVal {1}, a {2}, b {3}, str {4}".format(start, lastVal, a, b, str))
        if start == n: return True 
        result = False
        for i, element in enumerate(s, start=start):
            end = i+1
            tmpInt = int(num[start:end])
            if counter >1 and tmpInt < lastVal:
                continue
        
            s = num[end:]
            if s:
                if counter > 1 and s[0] == '0': return False
                if not a is None and not b is None:
                    if not tmpInt == a + b: return False  
            else:
                if a is None or b is None:
                    return False
                if not tmpInt == a + b: return False
            result = result or dfs(counter+1, i+1, tmpInt, b, tmpInt, num[end:])
            if result == True:
                return True
        return result 
    
    result = dfs(s=num)
    return result

#num = "112358"
# num = "199100199"
#num = "111"
#num = "101"
#num = "000"
# num = "1023"
# result =isAdditiveNumber(num) 
# print("Result is {0}".format(result)) 

def generateParenthesis(n: int):

    result = []   
    dp = {} 
    def dfs(open, close, s=""):
        if open > n: return
        if close > n: return
        
        if open == close: 
            dfs(open+1, close, s+'(')
       
        if open > close:
            dfs(open+1, close, s+'(')
            dfs(open, close+1, s+')')
        
        if len(s) == n*2:
            result.append(s)
        return
        
    dfs(0, 0)
    print(result)
    return result    
# n = 8
# result =generateParenthesis(n) 
# print("Result is {0}".format(result)) 

def diffWaysToCompute(expression: str):
    operators = ['*','+', '-']        
    def operandsAndsOperators():
        operands = []    
        i = 0
        n = len(expression)
        tmp = []
        ops = []
        while True:
            if i == n or expression[i] in operators :
                operands += [int("".join(tmp))]
                tmp = []
                if i == n:
                    break    
                ops += [expression[i]]  
            else:
                tmp += expression[i]
            i+=1
        return operands, ops
    operands, ops = operandsAndsOperators()
    n = len(ops)
    # visited = [False] * n
    print(operands, ops)

    def dfs(sum=0, j=0):
        if j == n: return 0
        result = 0
        for i, op in enumerate(ops, start=j):
            result += operands[i]+op+operands[i+1]+dfs(sum, i+1)
            result += operands[i]+op+dfs() 
        
        
                
    result = dfs()
    return result

# expression = "245-12-1"
# result = diffWaysToCompute(expression)
# print("Result is {0}".format(result)) 

def splitString(s: str) -> bool:
    
    n = len(s)
    badValues = []
    goodValues = []
    def dfs(start=0, prev=0, call=0):
        if start == n: return False
        leading = True
        result = False
        
        for i in range(start, n):
            while leading and i < n and s[i] == '0':
                i+=1
            leading = False
            end = i+1
            curr = int(s[start:end])
            print(prev , curr)
            result = result or (True and dfs(i+1, curr, call+1))
            if call == 0: return True
            if curr != prev - 1:
                #badValues.append(curr)
                return False
            goodValues.append(curr)
            result = True
            # print(result)
        return result
            
    result = dfs()
    print(goodValues, badValues)
    return result

#s = "4321"
#s = "050043"
# s = "9080701"
# # s = "0701"
# result = splitString(s)     
# print("Result is {0}".format(result)) 

def maxDepthAfterSplit(seq: str):
    stack  = []
    result = []
    depth = 0
    for char in seq:
        if char == '(':
            if len(stack) > 0:
                depth += 1
            stack.append(char) 
            result.append(depth)
        else:
            stack.pop()     
            result.append(depth)
            if len(stack) > 0:
                depth-=1
    return result
    
# seq = "(()())"
# seq = "()(())()"
# result = maxDepthAfterSplit(seq)     
# print("Result is {0}".format(result)) 

def reverseParentheses(s: str) -> str:
    stack = []
    result = []
    tmp = []
    depth = 0
    for char in s:
        if char == '(':
            depth += 1
            stack.append(char)    
        elif char == ')':
            tmp = []
            if depth > 0:
                depth -= 1
                while stack and stack[-1] != '(':
                    tmp.append(stack.pop())
                # the last '('
                stack.pop()
            
            if  depth > 0:    
                while tmp:
                    stack.append(tmp.pop(0))
            else:
                result += tmp
        
        else:
            if stack: 
                stack.append(char)
            else:
                result.append(char)        
    return "".join(result)    
            
#s = "(abcd)"
# s = "(u(love)i)"
#s = "(ed(et(oc))el)"
#s = "a(bcdefghijkl(mno)p)q"
# result = reverseParentheses(s)     
# print("Result is {0}".format(result)) 
def scoreOfParentheses(s: str) -> int:
    result = 0
    stack = []
    lastChar = ""
    n = len(s)
    for i, char in enumerate(s):
        if char == '(':
            stack.append('(')   
        else:
            stack.pop()
            if lastChar == '(':
                result += 1
            else:
                result *= 2
        lastChar = char
    return result

# s = "()()"
# s = "((()()))()"
# s = "(()(()))"
# result = scoreOfParentheses(s)
# print("Result is {0}".format(result)) 


class StockSpanner:

    def __init__(self):
        self.stack = []      
        
    def next(self, price) -> int:
        if len(self.stack) == 0:
            self.stack.append([self.i, price])
            return 1
        
        if price < self.stack[-1][1]: 
            self.stack.append([1, price])    
            return 1
        
        count = 0
        while self.stack and price > self.stack[-1][1]:
            count += self.stack[-1][0]
            self.stack.pop()
        
        self.stack.append([1+count, price])
        return 1 + count
    
# stockSpanner = StockSpanner()
# result = stockSpanner.next(100)
# print(result)

# stockSpanner.next(80)
# print(result)

# price = stockSpanner.next(60)
# print(price)

# result = stockSpanner.next(70)
# print(result)

# result = stockSpanner.next(60)
# print(result)

# result = stockSpanner.next(75)
# print(result)

# result = stockSpanner.next(85)
# print(result)


def nextGreaterElements(nums):
    stack = []
    n = len(nums)
    res= [0] * n
    for i in range(2*n-1, -1, -1):
        while stack and nums[stack[-1]] <= nums[i % n]:
            stack.pop()
        if len(stack) == 0:
            res[i % n] = -1 
        else:
            res[i % n] = nums[stack[-1]]
        stack.append(i % n)
    return res


# nums = [1,2,3,4,3]
# result = nextGreaterElements(nums)
# print("Result is {0}".format(result)) 

def exclusiveTime(n: int, logs):
    
    result = [0] * n
    for log in logs:
        id, status, timeStamp = log.split(":")
        print(id, status, timeStamp)
n = 2
# logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
# result = exclusiveTime(n, logs)
# print("Result is {0}".format(result)) 
def maxWidthRamp(num):
    stack = []
    result = 0
    map = {}
    maxWidth = 0 
    
    import math
    for i, num in enumerate(num):
        if len(stack) == 0: 
            stack.append([i, num])
            map[num] = i
            continue
        
        if num < stack[-1][1]:
            if map.get(num):
                stack.append([map[num], num])
            else:
                stack.append([i, num])
                map[num] = i
        else:
            least = math.inf
            while stack and num > stack[-1][1]:
                least =  min(least, stack[-1][0])
                stack.pop()
            stack.append([least, num])
            maxWidth = max(maxWidth, i-least)

    return maxWidth

# nums = [6,0,8,2,1,5]
# nums = [9,8,1,0,1,9,4,0,4,1]
# result = maxWidthRamp(nums)
# print("Result is {0}".format(result))

def singleNonDuplicate(nums):
    low = 0
    high = len(nums)-2
    import math
    while (low<=high):
        mid = (low+high)//2
        
        if nums[mid] == nums[mid^1]:
            low = mid+1
        else:
            high = mid-1
    return nums[low]

# nums = [1,1,2,3,3,4,4,8,8]
# result = singleNonDuplicate(nums)
# print("Result is {0}".format(result))

def searchRange(nums, target: int):
    n = len(nums)
    lx, rx = 0, n-1
    l, r = nums[0], nums[n-1]
    result = []
    while l <= r:
        mid = l + (r - l)//2
        mx = lx + (rx - lx) // 2
        
        if mid == target:
            result.append(mx)
            mx += 1        
            while nums[mx] == target:
                mx += 1            
            result.append(mx-1)
            break
        
        if mid < target:   
            l = mid + 1
            lx = mx + 1
        elif mid > target:
            r = mid - 1
            rx = mx - 1
    
    if result == []: 
        return [-1, -1]
    return result
    
    

# nums = [5,7,7,8,8,10]
# target = 8
# nums = [5,7,7,8,8,10]
# target = 6
# result = searchRange(nums, target)
# print("Result is {0}".format(result))


def getMaximumXor(nums, maximumBit: int):
    maxVal = 2 ** maximumBit - 1
    n = len(nums)
    prefix = [0] * n 
    result = [0] * n 
    sum = 0
    for i in range(n):
        sum = nums[i] ^ sum
        prefix[i] = sum
    
    for i in range(n-1, -1, -1):
        result[n-i-1] = prefix[i] ^ maxVal
    
    return result

# nums = [0,1,1,3]
# maximumBit = 2
# result = getMaximumXor(nums, maximumBit)
# print("Result is {0}".format(result))

def countTriplets(arr):
    n = len(arr)
    result = 0
    for i in range(0, n-1):
        prefix = 0 ^ arr[i]
        for j in range(i+1, n):
            prefix = prefix ^ arr[j]
            if prefix == 0:
                result += j - i
    return result
    
# arr = [2,3,1,6,7]
# arr = [1,1,1,1,1]
# result = countTriplets(arr)
# print("Result is {0}".format(result))


def largestMagicSquare(grid) -> int:
    m, n = len(grid), len(grid[0])
    ps = [[[0, 0, 0, 0] for _ in range(n + 2)] for _ in range(m + 2)]
    largest_size = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # update prefix sum
            ps[i][j][0] = ps[i][j - 1][0] + grid[i - 1][j - 1]
            ps[i][j][1] = ps[i - 1][j][1] + grid[i - 1][j - 1]
            ps[i][j][2] = ps[i - 1][j - 1][2] + grid[i - 1][j - 1]
            ps[i][j][3] = ps[i - 1][j + 1][3] + grid[i - 1][j - 1]
            
            # find the largest possible square
            k = min(i, j)
            while k:
                r, c = i - k, j - k
                k -= 1
                # check all the sum of row, col, diagonal is equal using set()
                s = set()
                # add top-left to bot-right diagonal to set
                s.add(ps[i][j][2] - ps[r][c][2])
                # add bot-left to top-right diagonal to set
                s.add(ps[i][j - k][3] - ps[i - k - 1][j + 1][3])
                if len(s) > 1:
                    continue
                    
                # add sum of each row to set
                for x in range(r + 1, i + 1):
                    s.add(ps[x][j][0] - ps[x][c][0])
                    
                if len(s) > 1:
                    continue
                
                # add sum of each col to set
                for x in range(c + 1, j + 1):
                    s.add(ps[i][x][1] - ps[r][x][1])
                    
                if len(s) > 1:
                    continue
                    
                largest_size = max(largest_size, k + 1)
                break
    return largest_size

# grid = [[7,1,4,5,6],[2,5,1,6,4],[1,5,4,3,2],[1,2,7,3,4]]
# result = largestMagicSquare(grid) 
# print("Result is {0}".format(result))

def numSubarraysWithSum(nums, goal: int):
    from collections import Counter
    P = [0]
    for x in nums:
        P.append(P[-1] + x)
    
    count = Counter()
    ans = 0
    for x in P:
        ans += count[x]
        count[x + goal] += 1
    print(count, P) 
    return ans


# nums = [1,0,1,0,1]
# goal = 2
# # nums = [0,0,0,0,0]
# # goal = 0
# result = numSubarraysWithSum(nums, goal)
# print("Result is {0}".format(result))


def goodDaysToRobBank(security, time: int):
    n = len(security)
    NI = [0] * n
    ND = [0] * n

    for i in range(1, n):
        if security[i] <= security[i-1]:
            NI[i] = 1
            if i - 1 == 0:
                NI[i-1] = 1
    
    for i in range(1, n):
        if security[i] >= security[i-1]:
            ND[i] = 1
            if i - 1 == 0:
                ND[i-1] = 1
        
    result = []
    for i in range(time, n-time):
        if NI[i-time] == 1 and ND[i+time] == 1:
            result.append(i)
        
    print(NI, ND)
    return result
    
# security = [5,3,3,3,5,6,2]
# time = 2
# security = [1,1,1,1,1]
# time = 0
# security = [1,2,3,4,5,6]
# time = 2
# security = [1,2,3,4]
# time = 0
# result = goodDaysToRobBank(security, time)
# print("Result is {0}".format(result))

def splitPainting(segments):
    segments = sorted(segments, key= lambda x: x[0]) 
    minTime = segments[0][0]
    maxTime = segments[-1][1]

    result = []
    for time in range(minTime, maxTime + 1):
        for segment in segments:
            s, e, d = segment
            
    
# segments = [[1,4,5],[4,7,7],[1,7,9]]
# # segments = [[1,7,9],[6,8,15],[8,10,7]]
# result = splitPainting(segments)
# print("Result is {0}".format(result))

def intervalIntersection(firstList, secondList):
    firstList = sorted(firstList, key=lambda x: x[0])
    secondList = sorted(secondList, key= lambda x: x[0])
    result = []
    for i, val1 in enumerate(firstList):
        s1, e1 = val1
        for j, val2 in enumerate(secondList):
            s2, e2 = val2
         
            if s2 <= e1:
                start = max(s1, s2)
                end = min(e1, e2)
                if end >= start:
                    intersect = [start, end]
                    result.append(intersect)
    return result    

# firstList = [[0,2],[5,10],[13,23],[24,25]]
# secondList = [[1,5],[8,12],[15,24],[25,26]]
# firstList = [[1,3],[5,9]]
# secondList = []
# firstList = [[0,5],[12,14],[15,18]]
# secondList = [[11,15],[18,19]]
# result = intervalIntersection(firstList, secondList)
# print("Result is {0}".format(result))

def merge(intervals):
    merged = []
    result = []
    merged = []
    n = len(intervals)
    mergeDone = False
    
    i = 0
    while i < n - 1:
        
        s1, e1 =  intervals[i] if not mergeDone else merged
        s2, e2 = intervals[i+1]
        mergeDone = False
        if s2 <= e1:
            mergeDone = True
            lo = min(s1, s2)
            hi = max(e1, e2)
            merged = [lo, hi]
            if result != []:
                result.pop()
            result.append(merged)
            i += 1
            continue
        else:
            mergeDone = False
            result.append(intervals[i])
            if i == n-2:
                result.append(intervals[i+1])
        i+=1
            
    
    return result
    
# intervals = [[1,3],[2,6],[8,14],[13,18]]
# intervals = [[1,4],[4,5]]
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# result = merge(intervals)
# print("Result is {0}".format(result))

def minimumRefill(plants, capacityA: int, capacityB: int):
    n = len(plants)
    l, r = 0, n-1
    global result
    result = 0
    sumA = capacityA
    sumB = capacityB
    Alice, Bob = 0, 1
    
    def capacity(sum, plant, AliceOrBob):
        global result
        if sum >= plant:
            sum -= plant
            return sum 
        
        result += 1
        if AliceOrBob == 0:
            sum = capacityA - plant
        else:
            sum = capacityB - plant
        return sum
        
    
    while (l < r):
        sumA = capacity(sumA, plants[l], Alice)
        sumB = capacity(sumB, plants[r], Bob)
        l += 1
        r -= 1
        if l == r:
            if sumA >= sumB:  
                sumA = capacity(sumA, plants[l], Alice)
            else:
                sumB = capacity(sumB, plants[r], Bob)      
            break
    return result
        
# plants = [2,2,3,3]
# capacityA = 5
# capacityB = 5
# plants = [2,2,3,3]
# capacityA = 3
# capacityB = 4
# plants = [5]
# capacityA = 10
# capacityB = 8
# plants = [2,2,5,2,2]
# capacityA = 5
# capacityB = 5
# plants = [7,7,7,7,7,7,7]
# capacityA = 8
# capacityB = 7
# plants = [2,1,1]
# capacityA = 2
# capacityB = 2
# result = minimumRefill(plants, capacityA, capacityB)
# print("Result is {0}".format(result))

def maxSumTwoNoOverlap(nums, firstLen: int, secondLen: int):
    n = len(nums)
    ps = [0]*(n+1)
    
   
# nums = [0,6,5,2,2,5,1,9,4]
# firstLen = 1
# secondLen = 2
# result = maxSumTwoNoOverlap(nums, firstLen, secondLen)
# print("Result is {0}".format(result))

def numberOfSubarrays(nums, k: int):
    n = len(nums)
    values = list(map(lambda x: 0 if x % 2 == 0 else 1, nums))
    print(values)    


# nums = [1,1,2,1,1]
# k = 3
# nums = [2,2,2,1,2,2,1,2,2,2]
# k = 2
# result = numberOfSubarrays(nums,k)
# print("Result is {0}".format(result))

def rangeSum(nums, n: int, left: int, right: int):
    n = len(nums)
    result = []
    for i in range(n):
        sum = nums[i]
        result.append(sum)
        for j in range(i+1, n):
            sum += nums[j]
            result.append(sum)
    
    result = sorted(result)

    ans = 0
    for i in range(left-1, right):
        ans = ans % (10**9 + 7) + result[i] % (10**9 + 7)
    return ans 

# nums = [1,2,3,4]
# n = 4
# left = 1
# right = 5
# result = rangeSum(nums, n, left, right)
# print("Result is {0}".format(result))

def countPalindromicSubsequence(s: str) -> int:
    from collections import defaultdict
    def helper(subS, s):
        i, j = 0, 0
        while i < 3 and j < n: 
            if subS[i] == s[j]:
                i += 1
            j += 1
            
        return i == 3

    counter = defaultdict(lambda: 0)
    for char in s: 
        counter[char] += 1
        
    candidates = set()
    for char1 in counter: 
        if counter[char1] >= 2: 
            for char2 in counter:
                candidates.add(char1 + char2 + char1)
            
    ans = 0
    n = len(s)
    
    for candidate in candidates: 
        if helper(candidate, s):
            ans += 1
            
    return ans   

# s = "aabca"
# result = countPalindromicSubsequence(s)
# print("Result is {0}".format(result))

def numSubarraysWithSum(nums, goal: int) -> int:
    from collections import Counter
    P = [0]
    for num in nums:
       P.append(P[-1] + num)
    
    count = Counter()
    ans = 0
    for x in P:
        ans += count[x]
        count[x+goal] += 1
    
    print(P, count)
    return ans
    
# nums = [1,0,1,0,1]
# goal = 2
# result = numSubarraysWithSum(nums, goal)
# print("Result is {0}".format(result))
    
def longestWPI(hours) -> int:   
    from collections import Counter
    ps = [0] + list(map(lambda x: 1 if x > 8 else -1, hours))
    for i in range(1, len(ps)):
        ps[i] = ps[i] + ps[i-1]


    d = Counter()
    longest = 0
    for i, val in enumerate(ps):
        if val > 0:
            longest = i
        else:
            if d.__contains__(val): 
                longest = max(longest, i-d[val])
            if not d.__contains__(val+1):
                d[val+1] = i
    
    return longest

    
#hours = [9,9,6,0,6,6,9]
# hours = [6,6,9]
# hours = [6,9,9]
# hours = [9,9,9]
# result = longestWPI(hours)
# print("Result is {0}".format(result))

def checkSubarraySum(nums, k: int) -> bool:
    n = len(nums)
    if n < 2: return False
    ps = [0] * n
    ps[0] = nums[0]
    for i in range(1, n):
        ps[i] = ps[i-1] + nums[i] 

    def bsearch(left, target):
        l, r = left, n-1
        while l <= r :
            mid = l + (r-l) // 2
            if ps[mid] == target: return True
            if ps[mid] < target:
                l = mid + 1
            if ps[mid] > target:
                r = mid - 1
        return False

    # watch for zeros
    for i in range(n):
        counter = 0
        while i <n and nums[i] == 0:
            counter += 1
            if counter > 1:
                return True
            i += 1
            

    for i, num in enumerate(ps):
        if num > 0 and num % k == 0: return True
        multiples = (ps[-1] - num) // k + 1 
        for multiple in range(1, multiples):
            target = nums[i] + multiple*k
            if bsearch(i, target):
                return True
            
    return False
        
   
# nums = [23,2,6,4,7]
# k = 6 
# nums = [23,2,6,4,7]
# k = 6
# nums = [23,2,6,4,7]
# k = 13
# nums = [23,2,4,6,6]
# k = 7
# nums = [5,0,0,0]
# k = 3
# nums = [1,0,1,0,1]
# k = 4
# nums = [0,1,0,3,0,4,0,4,0]
# k = 5
# nums = [1,2,3]
# k = 5
# nums = [1,2,12]
# k = 6
# nums = [23,6,9]
# k = 6
# result = checkSubarraySum(nums, k)
# print("Result is {0}".format(result))

=======
from operator import indexOf

from numpy import Inf

def minSwaps(s: str) -> int:
    opened = 0
    closed = 0
    n = len(s) 
    last = n-1
    s = [letter for letter in s]
   
    def swap(last):
        for j in range(last, -1, -1):
            if s[j] == '[':
                s[j] = ']'
                last = j-1
                return
    
    counter = 0
    for i, letter in enumerate(s):
        opened += 1 if (letter == '[') else 0
        closed += 1 if (letter == ']') else 0  
        if closed > opened:
            s[i] = '['
            swap(last)
            opened += 1
            closed -= 1
            counter += 1
    return counter
      
def minRemoveToMakeValid(s: str) -> str:
    s = [letter for letter in s]
    o = 0
    c = 0
    n = len(s)
    # forward
    for i, letter in enumerate(s):
        if letter == '(':
            o += 1
        if letter == ')':
            c += 1
            if c > o:
                s[i] = ""
                c -= 1
    
    o = 0
    c = 0
    # backward
    for i in range(n-1, -1 ,-1):
        if s[i] == ')':
            c += 1
        if s[i] == '(':
            o += 1
            if o > c:
                s[i] = ""
                o -= 1
    
    return "".join(s)


def hIndex(citations) -> int:
    # import heapq
    # h = []
    # for c in citations:
    #     heapq.heappush(h, -c)
    
    # ans = 0

    # while h:
    #     x = heapq.heappop(h)
    #     x = -x
    #     if x > ans:
    #         ans += 1
    
    # return ans
    dic = {}
    cit_sorted = sorted(citations)
    for i in range(len(cit_sorted)):
        if cit_sorted[i] not in dic:
            dic[cit_sorted[i]] = len(cit_sorted[i:])
        h_index = 0
    
    print(dic)
    for key in dic:
        if dic[key] >= h_index:
            if dic[key] < key:
                h_index = dic[key]
            else:
                h_index = key
    return h_index

def wiggleSort(nums) -> None:
    nums = sorted(nums)
    n = len(nums)
    mid = n // 2 if n % 2 == 0 else n // 2 + 1 
    
    lower = [0]*mid
    upper = [0]*(n - mid) 
    
    j = 0
    for i, value in enumerate(nums):
        if i < mid:
            lower[i] = value
        else:
            upper[j] = value
            j += 1
    
    nums = [0] * n
    for i in range(len(lower)):
        nums[2*i] = lower[i]
        if (2*i + 1) < n:
            nums[2*i + 1] = upper[i] 
    
    return nums

def canPartitionKSubsets(nums, k: int) -> bool:
    import functools
    sum = functools.reduce(lambda x, y: x + y, nums)
    target = sum//k
    n = len(nums)
    print(n, target, nums)
    visited = [False] * n
    dp = {}

    global count
    count = 0
    
    def dfs(j, sum):
        visited[j] = True
        if sum == 0: return 1
        if dp.get(sum): return dp[sum]
        
        count = 0
        for i in range(j+1, n):
            if sum - nums[i] >= 0:
                count += dfs(i, sum-nums[i])
                dp[sum] = count
        return count     
    
    result = 0
    for i in range(n):
        if not visited[i]:    
            if nums[i] == target:
                # if dp.get(nums[i]):
                #     dp[nums[i]] += 1
                result += 1
            else:
                result += dfs(i, target-nums[i])
                                    
    print(n, k, dp, result, visited)
       

def wordBreak(s: str, wordDict) -> bool:
    dp = {}
    for word in wordDict:
        dp[word] = False

    def dfs(s):
        if s == "": return True

        result = False    
        for word in wordDict:
            index = s.find(word)
            if not result and index == 0:
                if dp[word]: return True
                dp[word] = True
                news = s.removeprefix(word)
                if news == "": return True
                result = True and dfs(news)
        return result
    result = dfs(s)
    print(dp, result)
    return result
    
class RLEIterator:
    def __init__(self, encoding):
        self.encoding = []
        n = len(encoding)
        i = 0
        while i < n:
            self.encoding += [encoding[i+1]]*encoding[i]
            i += 2
        print(self.encoding) 

    def next(self, n: int) -> int:
        if len(self.encoding) < n:
            return -1 
        for _ in range(n):
            last = self.encoding.pop(0)
        return last
        
#s = "][]["
# s = "]]][[["
# result =  minSwaps(s)
# print("Result is {0}".format(result)) 

#s = "lee(t(c)o)de)"
#s = "))(("
#s = "(a(b(c)d)"
# result = minRemoveToMakeValid(s)
# print("Result is {0}".format(result)) 
# citations = [3,0,6,1,5]
# result = hIndex(citations)
# print("Result is {0}".format(result)) 

# nums = [1,5,1,1,6,4]
#nums = [1,2,1,3,2]
# result = wiggleSort(nums)
# print("Result is {0}".format(result)) 

# rLEIterator = RLEIterator([3, 8, 0, 9, 2, 5])
# result = rLEIterator.next(2)
# print(result) 
# result = rLEIterator.next(1)
# print(result)
# result = rLEIterator.next(1)
# print(result)
# result = rLEIterator.next(2)
# print(result)

# nums = [4,3,2,3,5,2,1]
# k = 4

# nums = [1,2,3,4]
# k = 3
# nums = [1,1,1,1,2,2,2,2]
# k = 4
# nums = [1,2,3,5]
# k = 2
# nums = [4,5,9,3,10,2,10,7,10,8,5,9,4,6,4,9]
# k = 5
# result = canPartitionKSubsets(nums, k)
# print("Result is {0}".format(result)) 


# s = "leetcode"
# wordDict = ["leet","code"]
# s = "applepenapple"
# wordDict = ["apple","pen"]
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# s = "cars"
# wordDict = ["car","ca","rs"]
# wordDict = ["ca","rs", "car"]
# s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
# wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
# s = "aaaaaaa"
# wordDict = ["aaaa","aa"]
# result = wordBreak(s, wordDict)
# print("Result is {0}".format(result)) 

def dailyTemperatures(temperatures):
    from collections import deque
    n = len(temperatures)
    result = [0]*n
    stack = []
    l = -1
    for i, currTmp in enumerate(temperatures):           
        while len(stack) > 0 and stack[l][1] < currTmp:
            index, _ = stack.pop()
            result[index] = i - index
            l -= 1
        stack.append((i, currTmp))
        l += 1
    return result     
        
# temperatures = [73,74,75,71,69,72,76,73]
# #temperatures = [30,60,90]
# result = dailyTemperatures(temperatures)
# print("Result is {0}".format(result)) 

def generateParenthesis(n: int):
    def dfs(n):
        if n == 0: return ""
        s = ""
        s += dfs(n-1)
        s += '(' +')'
        return s
    result = dfs(n)
    return result
# n = 4
# result = generateParenthesis(n)
# print("Result is {0}".format(result)) 

def numSubseq(nums, target) -> int:     
    nums = sorted(nums)
    counter = 0
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if j >= i and a + b <= target:
                counter += 1
                counter  = counter % ((10**9)+7) 
    return counter
# nums = [3,5,6,7]
# target = 9
# result = numSubseq(nums, target)
# print("Result is {0}".format(result)) 

def myPow(x: float, n: int) -> float:
    import math
    dp = {}
    def dfs(n):
        if n == 0: return 1
        if dp.get(n): return dp[n]
        if n < 0:
            dp[n] = 1/x * dfs(n+1)
        else:
            dp[n] = x * dfs(n-1)
        return dp[n]
    
    result = dfs(n)
    print(dp)                                     
    return result
    
    
# x = 2.00000
# n = 10
# x = 2.10000
# n = 3
# x = 2.00000
# n = -2
# result = myPow(x, n)
# print("Result is {0}".format(result)) 

def removeKdigits(num: str, k: int) -> str:
    n = len(num)
    visited = [False] * n

    # def convert2Num(s: str):

    def dfs(i, count=0):
        if count < k:
            visited[i] = True
        
        
        for j, _ in enumerate(num, start=i):
            if count < k-1:
                dfs(j+1, count+1)    
            
            tmp = ""
            for v, value in enumerate(visited):
                if not value: 
                    tmp += num[v]
            print(tmp)                  
            visited[i] = False
            return
    dfs(0)       
# num = "1432219"
# k = 3
# result =removeKdigits(num, k) 
# print("Result is {0}".format(result)) 

def findLeastNumOfUniqueInts(arr, k: int) -> int:
    n = len(arr)
    visited = [False] * n
    import math

    def returnLen():
        s = set()
        for v in range(n):
            if not visited[v]:
                s.add(arr[v])
        return len(s)
    
    def dfs(i, k):
        if k < 0: return 0  
        l = 0
        result = math.inf
        for j in range(i, n):
            visited[j] = True
            result = min(result, dfs(j+1, k-1))        
            visited[j] = False
            l = returnLen()
            if k == 0:
                result = l
        return result
    
    try:       
        result = dfs(0, k)
    except UnboundLocalError as e:
        print(e)
    return result

#arr = [1,2,3]        
#arr = [4,3,1,1]
# arr = [4,3,1,1,3,3,2]
# k = 3
#k = 2
#k = 2
# result =findLeastNumOfUniqueInts(arr, k) 
# print("Result is {0}".format(result)) 

def isAdditiveNumber(num: str) -> bool:
    
    if len(num) < 3: return False
    #if num[0] == '0': return False
    
    n = len(num)
    def dfs(counter=0, start=0, lastVal = 0, a= None, b=None, s="") -> bool:
        print("start {0}, lastVal {1}, a {2}, b {3}, str {4}".format(start, lastVal, a, b, str))
        if start == n: return True 
        result = False
        for i, element in enumerate(s, start=start):
            end = i+1
            tmpInt = int(num[start:end])
            if counter >1 and tmpInt < lastVal:
                continue
        
            s = num[end:]
            if s:
                if counter > 1 and s[0] == '0': return False
                if not a is None and not b is None:
                    if not tmpInt == a + b: return False  
            else:
                if a is None or b is None:
                    return False
                if not tmpInt == a + b: return False
            result = result or dfs(counter+1, i+1, tmpInt, b, tmpInt, num[end:])
            if result == True:
                return True
        return result 
    
    result = dfs(s=num)
    return result

#num = "112358"
# num = "199100199"
#num = "111"
#num = "101"
#num = "000"
# num = "1023"
# result =isAdditiveNumber(num) 
# print("Result is {0}".format(result)) 

def generateParenthesis(n: int):

    result = []   
    dp = {} 
    def dfs(open, close, s=""):
        if open > n: return
        if close > n: return
        
        if open == close: 
            dfs(open+1, close, s+'(')
       
        if open > close:
            dfs(open+1, close, s+'(')
            dfs(open, close+1, s+')')
        
        if len(s) == n*2:
            result.append(s)
        return
        
    dfs(0, 0)
    print(result)
    return result    
# n = 8
# result =generateParenthesis(n) 
# print("Result is {0}".format(result)) 

def diffWaysToCompute(expression: str):
    operators = ['*','+', '-']        
    def operandsAndsOperators():
        operands = []    
        i = 0
        n = len(expression)
        tmp = []
        ops = []
        while True:
            if i == n or expression[i] in operators :
                operands += [int("".join(tmp))]
                tmp = []
                if i == n:
                    break    
                ops += [expression[i]]  
            else:
                tmp += expression[i]
            i+=1
        return operands, ops
    operands, ops = operandsAndsOperators()
    n = len(ops)
    # visited = [False] * n
    print(operands, ops)

    def dfs(sum=0, j=0):
        if j == n: return 0
        result = 0
        for i, op in enumerate(ops, start=j):
            result += operands[i]+op+operands[i+1]+dfs(sum, i+1)
            result += operands[i]+op+dfs() 
        
        
                
    result = dfs()
    return result

# expression = "245-12-1"
# result = diffWaysToCompute(expression)
# print("Result is {0}".format(result)) 

def splitString(s: str) -> bool:
    
    n = len(s)
    badValues = []
    goodValues = []
    def dfs(start=0, prev=0, call=0):
        if start == n: return False
        leading = True
        result = False
        
        for i in range(start, n):
            while leading and i < n and s[i] == '0':
                i+=1
            leading = False
            end = i+1
            curr = int(s[start:end])
            print(prev , curr)
            result = result or (True and dfs(i+1, curr, call+1))
            if call == 0: return True
            if curr != prev - 1:
                #badValues.append(curr)
                return False
            goodValues.append(curr)
            result = True
            # print(result)
        return result
            
    result = dfs()
    print(goodValues, badValues)
    return result

#s = "4321"
#s = "050043"
# s = "9080701"
# # s = "0701"
# result = splitString(s)     
# print("Result is {0}".format(result)) 

def maxDepthAfterSplit(seq: str):
    stack  = []
    result = []
    depth = 0
    for char in seq:
        if char == '(':
            if len(stack) > 0:
                depth += 1
            stack.append(char) 
            result.append(depth)
        else:
            stack.pop()     
            result.append(depth)
            if len(stack) > 0:
                depth-=1
    return result
    
# seq = "(()())"
# seq = "()(())()"
# result = maxDepthAfterSplit(seq)     
# print("Result is {0}".format(result)) 

def reverseParentheses(s: str) -> str:
    stack = []
    result = []
    tmp = []
    depth = 0
    for char in s:
        if char == '(':
            depth += 1
            stack.append(char)    
        elif char == ')':
            tmp = []
            if depth > 0:
                depth -= 1
                while stack and stack[-1] != '(':
                    tmp.append(stack.pop())
                # the last '('
                stack.pop()
            
            if  depth > 0:    
                while tmp:
                    stack.append(tmp.pop(0))
            else:
                result += tmp
        
        else:
            if stack: 
                stack.append(char)
            else:
                result.append(char)        
    return "".join(result)    
            
#s = "(abcd)"
# s = "(u(love)i)"
#s = "(ed(et(oc))el)"
#s = "a(bcdefghijkl(mno)p)q"
# result = reverseParentheses(s)     
# print("Result is {0}".format(result)) 
def scoreOfParentheses(s: str) -> int:
    result = 0
    stack = []
    lastChar = ""
    n = len(s)
    for i, char in enumerate(s):
        if char == '(':
            stack.append('(')   
        else:
            stack.pop()
            if lastChar == '(':
                result += 1
            else:
                result *= 2
        lastChar = char
    return result

# s = "()()"
# s = "((()()))()"
# s = "(()(()))"
# result = scoreOfParentheses(s)
# print("Result is {0}".format(result)) 


class StockSpanner:

    def __init__(self):
        self.stack = []      
        
    def next(self, price) -> int:
        if len(self.stack) == 0:
            self.stack.append([self.i, price])
            return 1
        
        if price < self.stack[-1][1]: 
            self.stack.append([1, price])    
            return 1
        
        count = 0
        while self.stack and price > self.stack[-1][1]:
            count += self.stack[-1][0]
            self.stack.pop()
        
        self.stack.append([1+count, price])
        return 1 + count
    
# stockSpanner = StockSpanner()
# result = stockSpanner.next(100)
# print(result)

# stockSpanner.next(80)
# print(result)

# price = stockSpanner.next(60)
# print(price)

# result = stockSpanner.next(70)
# print(result)

# result = stockSpanner.next(60)
# print(result)

# result = stockSpanner.next(75)
# print(result)

# result = stockSpanner.next(85)
# print(result)


def nextGreaterElements(nums):
    stack = []
    n = len(nums)
    res= [0] * n
    for i in range(2*n-1, -1, -1):
        while stack and nums[stack[-1]] <= nums[i % n]:
            stack.pop()
        if len(stack) == 0:
            res[i % n] = -1 
        else:
            res[i % n] = nums[stack[-1]]
        stack.append(i % n)
    return res


# nums = [1,2,3,4,3]
# result = nextGreaterElements(nums)
# print("Result is {0}".format(result)) 

def exclusiveTime(n: int, logs):
    
    result = [0] * n
    for log in logs:
        id, status, timeStamp = log.split(":")
        print(id, status, timeStamp)
n = 2
# logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
# result = exclusiveTime(n, logs)
# print("Result is {0}".format(result)) 
def maxWidthRamp(num):
    stack = []
    result = 0
    map = {}
    maxWidth = 0 
    
    import math
    for i, num in enumerate(num):
        if len(stack) == 0: 
            stack.append([i, num])
            map[num] = i
            continue
        
        if num < stack[-1][1]:
            if map.get(num):
                stack.append([map[num], num])
            else:
                stack.append([i, num])
                map[num] = i
        else:
            least = math.inf
            while stack and num > stack[-1][1]:
                least =  min(least, stack[-1][0])
                stack.pop()
            stack.append([least, num])
            maxWidth = max(maxWidth, i-least)

    return maxWidth

# nums = [6,0,8,2,1,5]
# nums = [9,8,1,0,1,9,4,0,4,1]
# result = maxWidthRamp(nums)
# print("Result is {0}".format(result))

def singleNonDuplicate(nums):
    low = 0
    high = len(nums)-2
    import math
    while (low<=high):
        mid = (low+high)//2
        
        if nums[mid] == nums[mid^1]:
            low = mid+1
        else:
            high = mid-1
    return nums[low]

# nums = [1,1,2,3,3,4,4,8,8]
# result = singleNonDuplicate(nums)
# print("Result is {0}".format(result))

def searchRange(nums, target: int):
    n = len(nums)
    lx, rx = 0, n-1
    l, r = nums[0], nums[n-1]
    result = []
    while l <= r:
        mid = l + (r - l)//2
        mx = lx + (rx - lx) // 2
        
        if mid == target:
            result.append(mx)
            mx += 1        
            while nums[mx] == target:
                mx += 1            
            result.append(mx-1)
            break
        
        if mid < target:   
            l = mid + 1
            lx = mx + 1
        elif mid > target:
            r = mid - 1
            rx = mx - 1
    
    if result == []: 
        return [-1, -1]
    return result
    
    

# nums = [5,7,7,8,8,10]
# target = 8
# nums = [5,7,7,8,8,10]
# target = 6
# result = searchRange(nums, target)
# print("Result is {0}".format(result))


def getMaximumXor(nums, maximumBit: int):
    maxVal = 2 ** maximumBit - 1
    n = len(nums)
    prefix = [0] * n 
    result = [0] * n 
    sum = 0
    for i in range(n):
        sum = nums[i] ^ sum
        prefix[i] = sum
    
    for i in range(n-1, -1, -1):
        result[n-i-1] = prefix[i] ^ maxVal
    
    return result

# nums = [0,1,1,3]
# maximumBit = 2
# result = getMaximumXor(nums, maximumBit)
# print("Result is {0}".format(result))

def countTriplets(arr):
    n = len(arr)
    result = 0
    for i in range(0, n-1):
        prefix = 0 ^ arr[i]
        for j in range(i+1, n):
            prefix = prefix ^ arr[j]
            if prefix == 0:
                result += j - i
    return result
    
# arr = [2,3,1,6,7]
# arr = [1,1,1,1,1]
# result = countTriplets(arr)
# print("Result is {0}".format(result))


def largestMagicSquare(grid) -> int:
    m, n = len(grid), len(grid[0])
    ps = [[[0, 0, 0, 0] for _ in range(n + 2)] for _ in range(m + 2)]
    largest_size = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # update prefix sum
            ps[i][j][0] = ps[i][j - 1][0] + grid[i - 1][j - 1]
            ps[i][j][1] = ps[i - 1][j][1] + grid[i - 1][j - 1]
            ps[i][j][2] = ps[i - 1][j - 1][2] + grid[i - 1][j - 1]
            ps[i][j][3] = ps[i - 1][j + 1][3] + grid[i - 1][j - 1]
            
            # find the largest possible square
            k = min(i, j)
            while k:
                r, c = i - k, j - k
                k -= 1
                # check all the sum of row, col, diagonal is equal using set()
                s = set()
                # add top-left to bot-right diagonal to set
                s.add(ps[i][j][2] - ps[r][c][2])
                # add bot-left to top-right diagonal to set
                s.add(ps[i][j - k][3] - ps[i - k - 1][j + 1][3])
                if len(s) > 1:
                    continue
                    
                # add sum of each row to set
                for x in range(r + 1, i + 1):
                    s.add(ps[x][j][0] - ps[x][c][0])
                    
                if len(s) > 1:
                    continue
                
                # add sum of each col to set
                for x in range(c + 1, j + 1):
                    s.add(ps[i][x][1] - ps[r][x][1])
                    
                if len(s) > 1:
                    continue
                    
                largest_size = max(largest_size, k + 1)
                break
    return largest_size

# grid = [[7,1,4,5,6],[2,5,1,6,4],[1,5,4,3,2],[1,2,7,3,4]]
# result = largestMagicSquare(grid) 
# print("Result is {0}".format(result))

def numSubarraysWithSum(nums, goal: int):
    from collections import Counter
    P = [0]
    for x in nums:
        P.append(P[-1] + x)
    
    count = Counter()
    ans = 0
    for x in P:
        ans += count[x]
        count[x + goal] += 1
    print(count, P) 
    return ans


# nums = [1,0,1,0,1]
# goal = 2
# # nums = [0,0,0,0,0]
# # goal = 0
# result = numSubarraysWithSum(nums, goal)
# print("Result is {0}".format(result))


def goodDaysToRobBank(security, time: int):
    n = len(security)
    NI = [0] * n
    ND = [0] * n

    for i in range(1, n):
        if security[i] <= security[i-1]:
            NI[i] = 1
            if i - 1 == 0:
                NI[i-1] = 1
    
    for i in range(1, n):
        if security[i] >= security[i-1]:
            ND[i] = 1
            if i - 1 == 0:
                ND[i-1] = 1
        
    result = []
    for i in range(time, n-time):
        if NI[i-time] == 1 and ND[i+time] == 1:
            result.append(i)
        
    print(NI, ND)
    return result
    
# security = [5,3,3,3,5,6,2]
# time = 2
# security = [1,1,1,1,1]
# time = 0
# security = [1,2,3,4,5,6]
# time = 2
# security = [1,2,3,4]
# time = 0
# result = goodDaysToRobBank(security, time)
# print("Result is {0}".format(result))

def splitPainting(segments):
    segments = sorted(segments, key= lambda x: x[0]) 
    minTime = segments[0][0]
    maxTime = segments[-1][1]

    result = []
    for time in range(minTime, maxTime + 1):
        for segment in segments:
            s, e, d = segment
            
    
# segments = [[1,4,5],[4,7,7],[1,7,9]]
# # segments = [[1,7,9],[6,8,15],[8,10,7]]
# result = splitPainting(segments)
# print("Result is {0}".format(result))

def intervalIntersection(firstList, secondList):
    firstList = sorted(firstList, key=lambda x: x[0])
    secondList = sorted(secondList, key= lambda x: x[0])
    result = []
    for i, val1 in enumerate(firstList):
        s1, e1 = val1
        for j, val2 in enumerate(secondList):
            s2, e2 = val2
         
            if s2 <= e1:
                start = max(s1, s2)
                end = min(e1, e2)
                if end >= start:
                    intersect = [start, end]
                    result.append(intersect)
    return result    

# firstList = [[0,2],[5,10],[13,23],[24,25]]
# secondList = [[1,5],[8,12],[15,24],[25,26]]
# firstList = [[1,3],[5,9]]
# secondList = []
# firstList = [[0,5],[12,14],[15,18]]
# secondList = [[11,15],[18,19]]
# result = intervalIntersection(firstList, secondList)
# print("Result is {0}".format(result))

def merge(intervals):
    merged = []
    result = []
    merged = []
    n = len(intervals)
    mergeDone = False
    
    i = 0
    while i < n - 1:
        
        s1, e1 =  intervals[i] if not mergeDone else merged
        s2, e2 = intervals[i+1]
        mergeDone = False
        if s2 <= e1:
            mergeDone = True
            lo = min(s1, s2)
            hi = max(e1, e2)
            merged = [lo, hi]
            if result != []:
                result.pop()
            result.append(merged)
            i += 1
            continue
        else:
            mergeDone = False
            result.append(intervals[i])
            if i == n-2:
                result.append(intervals[i+1])
        i+=1
            
    
    return result
    
# intervals = [[1,3],[2,6],[8,14],[13,18]]
# intervals = [[1,4],[4,5]]
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# result = merge(intervals)
# print("Result is {0}".format(result))

def minimumRefill(plants, capacityA: int, capacityB: int):
    n = len(plants)
    l, r = 0, n-1
    global result
    result = 0
    sumA = capacityA
    sumB = capacityB
    Alice, Bob = 0, 1
    
    def capacity(sum, plant, AliceOrBob):
        global result
        if sum >= plant:
            sum -= plant
            return sum 
        
        result += 1
        if AliceOrBob == 0:
            sum = capacityA - plant
        else:
            sum = capacityB - plant
        return sum
        
    
    while (l < r):
        sumA = capacity(sumA, plants[l], Alice)
        sumB = capacity(sumB, plants[r], Bob)
        l += 1
        r -= 1
        if l == r:
            if sumA >= sumB:  
                sumA = capacity(sumA, plants[l], Alice)
            else:
                sumB = capacity(sumB, plants[r], Bob)      
            break
    return result
        
# plants = [2,2,3,3]
# capacityA = 5
# capacityB = 5
# plants = [2,2,3,3]
# capacityA = 3
# capacityB = 4
# plants = [5]
# capacityA = 10
# capacityB = 8
# plants = [2,2,5,2,2]
# capacityA = 5
# capacityB = 5
# plants = [7,7,7,7,7,7,7]
# capacityA = 8
# capacityB = 7
# plants = [2,1,1]
# capacityA = 2
# capacityB = 2
# result = minimumRefill(plants, capacityA, capacityB)
# print("Result is {0}".format(result))

def maxSumTwoNoOverlap(nums, firstLen: int, secondLen: int):
    n = len(nums)
    ps = [0]*(n+1)
    
   
# nums = [0,6,5,2,2,5,1,9,4]
# firstLen = 1
# secondLen = 2
# result = maxSumTwoNoOverlap(nums, firstLen, secondLen)
# print("Result is {0}".format(result))

def numberOfSubarrays(nums, k: int):
    n = len(nums)
    values = list(map(lambda x: 0 if x % 2 == 0 else 1, nums))
    print(values)    


# nums = [1,1,2,1,1]
# k = 3
# nums = [2,2,2,1,2,2,1,2,2,2]
# k = 2
# result = numberOfSubarrays(nums,k)
# print("Result is {0}".format(result))

def rangeSum(nums, n: int, left: int, right: int):
    n = len(nums)
    result = []
    for i in range(n):
        sum = nums[i]
        result.append(sum)
        for j in range(i+1, n):
            sum += nums[j]
            result.append(sum)
    
    result = sorted(result)

    ans = 0
    for i in range(left-1, right):
        ans = ans % (10**9 + 7) + result[i] % (10**9 + 7)
    return ans 

# nums = [1,2,3,4]
# n = 4
# left = 1
# right = 5
# result = rangeSum(nums, n, left, right)
# print("Result is {0}".format(result))

def countPalindromicSubsequence(s: str) -> int:
    from collections import defaultdict
    def helper(subS, s):
        i, j = 0, 0
        while i < 3 and j < n: 
            if subS[i] == s[j]:
                i += 1
            j += 1
            
        return i == 3

    counter = defaultdict(lambda: 0)
    for char in s: 
        counter[char] += 1
        
    candidates = set()
    for char1 in counter: 
        if counter[char1] >= 2: 
            for char2 in counter:
                candidates.add(char1 + char2 + char1)
            
    ans = 0
    n = len(s)
    
    for candidate in candidates: 
        if helper(candidate, s):
            ans += 1
            
    return ans   

# s = "aabca"
# result = countPalindromicSubsequence(s)
# print("Result is {0}".format(result))

def numSubarraysWithSum(nums, goal: int) -> int:
    from collections import Counter
    P = [0]
    for num in nums:
       P.append(P[-1] + num)
    
    count = Counter()
    ans = 0
    for x in P:
        ans += count[x]
        count[x+goal] += 1
    
    print(P, count)
    return ans
    
# nums = [1,0,1,0,1]
# goal = 2
# result = numSubarraysWithSum(nums, goal)
# print("Result is {0}".format(result))
    
def longestWPI(hours) -> int:   
    from collections import Counter
    ps = [0] + list(map(lambda x: 1 if x > 8 else -1, hours))
    for i in range(1, len(ps)):
        ps[i] = ps[i] + ps[i-1]


    d = Counter()
    longest = 0
    for i, val in enumerate(ps):
        if val > 0:
            longest = i
        else:
            if d.__contains__(val): 
                longest = max(longest, i-d[val])
            if not d.__contains__(val+1):
                d[val+1] = i
    
    return longest

    
#hours = [9,9,6,0,6,6,9]
# hours = [6,6,9]
# hours = [6,9,9]
# hours = [9,9,9]
# result = longestWPI(hours)
# print("Result is {0}".format(result))

def checkSubarraySum(nums, k: int) -> bool:
    n = len(nums)
    if n < 2: return False
    ps = [0] * n
    ps[0] = nums[0]
    for i in range(1, n):
        ps[i] = ps[i-1] + nums[i] 

    def bsearch(left, target):
        l, r = left, n-1
        while l <= r :
            mid = l + (r-l) // 2
            if ps[mid] == target: return True
            if ps[mid] < target:
                l = mid + 1
            if ps[mid] > target:
                r = mid - 1
        return False

    # watch for zeros
    for i in range(n):
        counter = 0
        while i <n and nums[i] == 0:
            counter += 1
            if counter > 1:
                return True
            i += 1
            

    for i, num in enumerate(ps):
        if num > 0 and num % k == 0: return True
        multiples = (ps[-1] - num) // k + 1 
        for multiple in range(1, multiples):
            target = nums[i] + multiple*k
            if bsearch(i, target):
                return True
            
    return False
        
   
# nums = [23,2,6,4,7]
# k = 6 
# nums = [23,2,6,4,7]
# k = 6
# nums = [23,2,6,4,7]
# k = 13
# nums = [23,2,4,6,6]
# k = 7
# nums = [5,0,0,0]
# k = 3
# nums = [1,0,1,0,1]
# k = 4
# nums = [0,1,0,3,0,4,0,4,0]
# k = 5
# nums = [1,2,3]
# k = 5
# nums = [1,2,12]
# k = 6
# nums = [23,6,9]
# k = 6
# result = checkSubarraySum(nums, k)
# print("Result is {0}".format(result))

>>>>>>> 6491f59943a7f2d401bac25b7dedf78025fc6ea7
