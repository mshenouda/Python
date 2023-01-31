import collections
from collections import Counter

def numberOfPairs(nums):
    c = Counter(nums);
    i = 0
    result = [0, 0]
    for k, v in c.items():
        if v > 1:
            result[0] += 1
            result[1] += max(result[1], v%2)
        else:
            result[1] += 1
    return result

#nums = [1,3,2,1,3,2,2]
# nums = [12,36,75,76,50,36,36]
# result = numberOfPairs(nums)
# print('Result is {}'.format(result))   
def canReach(s: str, minJump: int, maxJump: int) -> bool:
    if s[-1] == '1': return False
    dp = [False] * len(s)
    dp[0] = True
    cnt = 1
    for i in range(minJump, len(s)):
        if s[i] == '0' and cnt > 0:
            dp[i] = True
        # left side of window
        if i - maxJump >= 0 and dp[i - maxJump]:
            cnt-=1
        # right side of window
        if dp[i-minJump+1]:
            cnt+=1
    return dp[len(s)-1]

# s = "01101110"
# s = "011010"
# minJump = 2
# maxJump = 3
# result = canReach(s, minJump, maxJump);
# print('Result is {}'.format(result))   
def maxSumMinProduct(nums) -> int:
    prefix = nums.copy()
    prefix.insert(0,0)
    length = len(prefix)
    for i in range(1,length):
        prefix[i]+=prefix[i-1]
    
    stack = []
    res = 0
    
    for i,num in enumerate(nums):
        newStart = i
        while stack and stack[-1][0]>num:
            val,idx = stack.pop()
            newStart = idx
            res = max(res,val*(prefix[i]-prefix[idx]))
        stack.append([num,newStart])
    
    for val,idx in stack:
        res = max(res,val*(prefix[-1]-prefix[idx]))
        
    return res%(10**9+7)

# nums = [1,2,3,2]
# result = maxSumMinProduct(nums)
# print('Result is {}'.format(result))   
def numOfPairs(nums, tar: str) -> int:
    cnter, ans = Counter(nums), 0
    for n in cnter:
        if n+n==tar:
            ans += cnter[n]*(cnter[n]-1)
        elif tar[:len(n)]==n:
            ans += cnter[n]*cnter[tar[len(n):]]
    return ans
# nums = ["123","4","12","34"]
# target = "1234"
# result = numOfPairs(nums, target)
# print('Result is {}'.format(result))   

def start(METHODS):
    str.swapcase()
result = start("METHODS")
print(result)