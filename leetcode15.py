from heapq import heapify, heappush, heappop
from collections import Counter

dp = {0: 1}
def fab(n: int):
    if n < 0:
        raise ValueError(f"Only integers >= 0 are allowed")
    if n in dp: return dp[n]
    dp[n] = n*fab(n-1) 
    return dp[n]
    
# result = fab(50)
# print(f'The result is {result}')
    
def mergeList(ol, al, dl):
    s1 = set(ol)
    s2 = set(al)
    s3 = set(dl)
    s1.update(s2)
    s1.difference_update(s3)

    d = dict()
    for key in s1:
        d[key] = len(key) 
    d = sorted(d.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    result = [val[0] for val in d]
    return result

ol = ['one', 'two', 'three']
al = ['one', 'two', 'five', 'six']
dl = ['two', 'five']
# answer = mergeList(ol, al, dl)
# print(f'The result is {answer}')



def gibonnci(n,x, y):
    # dp = {0: x, 1: y, 2: y-x}
    # def recursive(n, x, y):
    #     if n in dp: return dp[n]
    #     if n % 2 == 0: return y - x
    #     return x - y
    # def recursive(n, x, y):
    #     if n in dp: return dp[n]
    #     dp[n] = recursive(n-1, x, y) - recursive(n-2, x, y)
    #     return dp[n]
    
    if n % 3 == 0 and (n // 3)%2 == 1: return -x 
    if n % 3 == 0 and (n // 3)%2 == 0: return x 
    if n % 3 == 1 and (n // 3)%2 == 1: return -y 
    if n % 3 == 1 and (n // 3)%2 == 0: return y
    if n % 3 == 2 and (n // 3)%2 == 1: return x-y 
    if n % 3 == 2 and (n // 3)%2 == 0: return y-x     
    
    
     
    # result = recursive(n, x, y)
    # print(dp)
    # return result
n = 7
# x = 7
# y= 8
# x = 3
# y= 5
x = 0
y = 1
# result = gibonnci(n, x, y)
# print(f'The result is {result}')

def check(n, m, games):
    s1 = set()
    s2 = set()
    for game in games:
        team1 = sorted(game[0: n//2])
        team2 = sorted(game[n//2: n])
        if team1[0] < team2[0]:
            s1.add(tuple(team1)) 
            s2.add(tuple(team2))
        else:
            s2.add(tuple(team1))
            s1.add(tuple(team2))
            
    if len(s1) == m or len(s1) == m-1: return True
    return False

n = 6 
m = 6
games = [[1,6,3,4,5,2], [6,4,2,3,1,5], [4,2,1,5,6,3], [4,5,1,6,2,3], [3,2,5,1,6,4], [2,3,6,4,1,5]]
# games = [[3,1,4,5,6,2], [5,3,2,4,1,6], [5,3,6,4,2,1], [6,5,3,2,1,4], [5,4,1,2,6,3], [4,1,6,2,5,3]]

# result = check(n, m, games)
# print(f'The result is {result}')

#print("%.2f" % 3.142)

# a = list(range(1, 11))
# print(a)
# b = a[0:-10]
# print(b)

def bar(x, *y, **z):
    print(x)
    print(y)
    print(z)

#bar(5,6,7, a=3, b=5)

def restoreIpAddresses(s: str):
    result = []
    
    def isValid(tmp, counter):
        if counter > 3:
            return False
        if tmp[0] == 0 and tmp[1] == 0:
            return False
        integer = int("".join([x for x in tmp]))
        if integer < 0 or integer > 255:
            return False
        return True
    
    def isComplete(index, counter):
        if index == len(s) and counter == 4:
            return True
        return False
            
    def dfs(start=0, counter=0, tmp=[]):
        for i in range(start, len(s)):
            tmp.append(s[i]);
            if isValid(tmp, counter):
                if isComplete(i, counter):
                    result.append(tmp)
            else:
                break                    
            dfs(i+1, counter+1, tmp)
            tmp.pop()
            
    dfs(0, 0)
    return result
    
#s = "25525511135"
# s = "25511"
# result = restoreIpAddresses(s)
# print(f'The result is {result}')

def kSmallestPairs(nums1, nums2, k: int):
    heapify(nums1)
    heapify(nums2)
    result = [[0,0] for i in range(k)]
    for i in range(k):
        if nums1:
            x = heappop(nums1)
        else:
            break
    
        if nums2:     
            y = heappop(nums2)
        else: 
            break
        heappush(nums1, x) if x <=y else heappush(nums2, y)
        result[i] = [x, y]
    return result

# nums1 = [1,7,11]
# nums2 = [2,4,6]
# k = 3
# nums1 = [1,1,2]
# nums2 = [1,2,3]
# k = 2
# nums1 = [1,2]
# nums2 = [3]
# k = 3
# result =  kSmallestPairs(nums1, nums2, k)
# print(f'The result is {result}')     

def countNumbersWithUniqueDigits(n: int) -> int:
    if n == 0: 
        return 1

    count = 0
    def dfs(start, k, tmp=set()):
        for i in range(start, 3):
            if k <= n:    
                tmp.add(i)
                if len(tmp) < k:
                    tmp.pop()
                    count = count + 1
                else:   
                    dfs(0, k+1)

    result = dfs(1, 1)
    print(f'The result is {count}')
    return result        
            
# n = 2
# result = countNumbersWithUniqueDigits(n)
# print(f'The result is {result}')     

def smallestNumber(pattern: str) -> str:
    l= len(pattern)
    
    def greater(tmp, i)-> int:
        if tmp == ():
            return 1   
        for j in range(1, 10):
            if j > tmp[-1]  :
                return j 
        return -1
    
    def smaller(i: int)-> int:
        for j in range(9, -1, -1):
            if j < i:
                return j
        return -1 
    
    def dfs(index: int, start, tmp= [1]):
        if index == l: return ()
        if pattern[index] == 'I':
            for i in range(start, 10):
                if not tmp.__contains__(i):
                    tmp.append(i)
                    dfs(index+1, start+1, tmp)
                else:
                    tmp.pop()
        else:
            for i in range(start, 10):
                if not tmp.__contains__(i):
                    tmp.append(i)
                    dfs(index+1, start, tmp)
                else:
                    tmp.pop()
                    
    result = dfs(0, 1)
    print(result)
      
#/pattern = "IIIDIDDD"
# pattern = "ID"
# result = smallestNumber(pattern)
# print(f'The result is {result}')     
class LUPrefix:

    def __init__(self, n: int):
        heapify(self.counter)
        
    def upload(self, video: int) -> None:
        heappush(self.counter, video)

    def longest(self) -> int:
        result = 0
        while self.counter:
            last = heappop(self.counter)
            if last == result+1:
                result = last  
            else:
                heappush(self.counter, last)    
        return result
    
def leastInterval(tasks, n: int) -> int:
    heap = []
    c = Counter(tasks)
    for k, v in c.items():
        heap.append([-v, k])
    heapify(heap)
    
    result = []
    while heap:
        last1 = heappop(heap)
        while last1[0] < 0:
            last1[0] += 1
            # heappush(heap, popped)
        
            # print(popped)
        
    print(heap)


# tasks = ["A","A","A","B","B","B"]
# n = 2
# result = leastInterval(tasks, n)
# print(f'The result is {result}')

def kSmallestPairs(nums1, nums2, k: int):
    h1 = nums1
    h2 = nums2
    heapify(h1)
    heapify(h2)
    result = []
    
    while h1 and h2:
        if len(h1) == 1:
            while h2 and len(result) < k:
                result.append([h2[0], h1[0]])    
                heappop(h2)
            break
        elif len(h2) == 1:
            while h1 and len(result) < k:
                result.append([h1[0], h2[0]])
                heappop(h1)
            break
        else:      
            result.append([h1[0], h2[0]])
            if len(result) == k:
                break
            if h1[0] + h2[1] < h1[1] + h2[0]: 
                heappop(h2)
            elif h1[0] + h2[1] > h1[1] + h2[0]:
                heappop(h1)
            elif h1[0] + h2[1] == h1[1] + h2[0]:
                result.append([h1[0], h2[1]])
                result.append([h1[1], h2[0]])
                if len(result) >= k:
                    break    
        
    return result
    
nums1 = [1,1,2]
nums2 = [1,2,3]
k = 2
# nums1 = [1,7,11]
# nums2 = [2,4,6]
# k = 3
# nums1 = [1,2]
# nums2 = [3]
# k = 3
# result = kSmallestPairs(nums1, nums2, k)
# print(f'The result is {result}')

def miceAndCheese(reward1, reward2, k: int) -> int:
    l1 = []
    l2 = []
    heapify(l1)
    heapify(l2)
    for i,e in enumerate(reward1):
        heappush(l1, (-e,i))
    for i,e in enumerate(reward2):
        heappush(l2, (-e,i))
    
    r1 = 0
    for _ in range(k):
        e, i  = heappop(l1)
        val1 = reward2[i]     
        r1 += e
        while l1[0][0] == e:
            j = l1[0][1]
            val2 = reward2[j]
            if val1 < val2:    
                r1 += -e
    for e, i in l1:
        r1 += reward2[i]
    print(l1)
    ind1 = [i[0] for i in sorted(enumerate(reward1), key= lambda x:x[1], reverse=True)]
    reward1 = sorted(reward1, reverse=True)
    print(ind1, reward1)
    reward1[4] = None
    print(reward1)
    
    
    r2 = 0
    for _ in range(k):
        e, i  = heappop(l2)     
        r2 += -e
    for e, i in l2:
        r2 += reward1[i]
    
    print(r1, r2)
    return max(r1 ,r2)

# reward1 = [4,1,5,3,3]
# reward2 = [3,4,4,5,2]
# k = 3
# result = miceAndCheese(reward1, reward2, k)
# print(f'The result is {result}')