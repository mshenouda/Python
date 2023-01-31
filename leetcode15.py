

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

bar(5,6,7, a=3, b=5)

5 not i