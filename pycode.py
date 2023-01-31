<<<<<<< HEAD

import re

def startFunc(first, second):
    print(f"Hello, {first}, i'm {second}")

def sfr(jobs, index):
    from heapq import heapify, heappush, heappop
    heap = []
    heapify(heap)
    for job in jobs:
        heappush(heap, job)

    sum  = 0   
    for _ in range(len(jobs)):
        item = heappop(heap)
        if item > jobs[index]:
            break
        sum += item
    return sum

def findMax(*args):
    return max(args)

def findSum(*args):
    return sum(args)

def reverseString(str):
    return str[::-1]

def findFactorial(num):
    if num <=  1:
        return 1
    return num * findFactorial(num -1)

def findLowerUpper(str):
   l = list(filter(lambda x: x>='a' and x <='z', str))
   u = list(filter(lambda x: x>='A' and x <='Z', str))
   return len(l), len(u)

def findMaxMin(l):
    return list(filter(lambda x: x % 2 == 1, l))

def findUnique(l):
    return set(l)

def findPalindrome(str):
    for i in range(len(str)):
        if str[i] != str[-1-i]:
            return False
    return True

def sortWords(str):
    return '-'.join(sorted(str.split('-')))

def findValues(n):
    return list(map((lambda x: x**2), range(1, n+1)))

def sortTubles(lst):
    return sorted(lst, key=(lambda x: x[1]))

def sortDict(lst):
    return sorted(lst, key=(lambda x: x['model']), reverse=True)

def filterLst(lst):
    funcs = [(lambda x: x % 2 == 0), (lambda x: x % 2 == 1)]
    for func in funcs:
        print(list(filter(func, lst)))
    funcs = [(lambda x: x ** 2), (lambda x: x ** 3)]
    for func in funcs:
        print(list(map(func, lst)))

def filterData(str):
    s = str.split(' ')
    return s[0].split('-') , s[-1]

def checkNum(lst):
    return list(filter(lambda x: isinstance(x, int) and (x != True and x != False), lst))

def findIntersect(lst1, lst2):
    lst = []
    for element in lst1:
        if element in lst2:
            lst.append(element)
    return lst

def arrange(lst):
    lst = sorted(lst)
    n = list(filter((lambda x: x < 0), lst))
    p = list(filter((lambda x: x >= 0), lst))
    return p + n

def add(lst1, lst2):
    pass

def findSecond(lst):
    s = sorted(set(map((lambda x: x[1]), lst)))
    m = list(filter(lambda x: x[1] == s[1], lst))
    return sorted(m, key=(lambda x: x[0]))

def findPalindromes(lst):
    def palindrome(str):
        for i in range(len(str)):
            if str[i] != str[-1-i]:
                return False
        return True
    return list(filter(lambda x: palindrome(x),lst)) 

def findAnagrams(lst, str):
    s = set(str)
    f = list(filter((lambda x: set(x) == s), lst))
    return f

def findNumbers(str):
    s = str.split(' ')
    l = []
    for number in s:
        try:
            tmp = int(number)
            l.append(tmp)
        except ValueError as e:
            pass
    return sorted(l)

def sumPN(lst):
    l = list(filter(lambda x: x<0, lst))
    p = list(filter(lambda x: x >=0, lst))
    return sum(l), sum(p)

def removeNone(lst):
    return list(filter((lambda x: x != None),lst))

def removeWords(lst, remove):
    l = []
    for num in lst:
        for rem in remove:
            if num == rem:
                break
        else:
            l.append(num)
    return l

def countOcc(lst):
    from collections import Counter
    return Counter(lst)

def sortStrs(lst):
    return sorted(lst, key=(lambda x: int(x)))

def findAvgs(lst):
    return list(map(lambda x: round(sum(x)/len(x), 2), lst))

def findMul(lst):
    from functools import reduce
    return reduce((lambda x, y: x * y), lst)

def reverseStrs(lst):
    return list(map((lambda x: x[::-1]),lst))

def find3(str):
    from collections import Counter
    c = Counter(str)
    return sorted(c.items(), key=(lambda item: item[1]), reverse=True)[0:3]

def mergeDict(dic1, dic2):
    return {**dic1, **dic2}

def mergeList(lst1, lst2):
    return [*lst1] + [*lst2]


def lis(lst):
    dp = [0] * len(lst)
    for i in range(len(lst)-1):
        for j in range(i+1, len(lst)):
            if lst[j] > lst[i] and dp[i] <= dp[j]:
                dp[j] = dp[i] + 1    
    
    print(dp)
    maxIndex = lst[dp.index(max(dp))]
    leastIndex = lst[dp.index(max(dp)) - max(dp)]
    return leastIndex, maxIndex

def reg(pattern, str):
    import re
    s = str.split('\n')
    res = re.search(pattern, str, re.MULTILINE)
    res = res.group('key')
    return res


def read_file(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def pattern2(pattern, str):
    result = re.findall(pattern, str)
    print(result)
    pass

if  __name__ == '__main__':
    #startFunc('Mina', 'Shenouda')
    # result = sfr([3, 10, 20, 1, 2], 0)
    # print(result)
    # result = sfr([2, 10, 10, 1, 3], 2)
    # print(result)
    # result = sfr([10, 10, 10, 10], 3)
    # print(result)
    # result = findMax(1, 2 , 3)
    # print(result)
    # result = findSum(1, 3, 6 , 8 , 9)
    # print(result)
    # result = reverseString('minaShenouda')
    # print(result)
    # result = findFactorial(5)
    # print(result)
    # result, result1 = findLowerUpper('The quick Brow Fox')
    # print(result, result1)
    # # result = findMaxMin([1, 5, 9, 12])
    # print(result)
    # result = findUnique([1, 2, 3, 3, 3 , 3 , 3])
    # print(result)
    # result = findPalindrome('cannac')
    # print(result)
    # result = sortWords('green-red-yellow-black-white')
    # print(result)
    # result = findValues(30)
    # print(result)
    # result = (lambda x: x + 15)
    # print(result(5))
    # result = (lambda x, y: x * y)
    # print(result(5, 15))
    # result = (lambda x: 15 * x)
    # print(result(4))
    # result = sortTubles([('English', 88), ('Science', 90), ('Maths', 97), ('Social sciences', 82)])
    # print(result)
    # result = sortDict([{'make': 'Nokia', 'model': 216, 'color': 'Black'}, {'make': 'Samsung', 'model': 7, 'color': 'Blue'}, {'make': 'Mi Max', 'model': 2, 'color': 'Gold'}])
    # print(result)
    # result = filterLst([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(result)
    # result = filterData('2020-01-15 09:03:32.744178')
    # print(result)
    # result = checkNum([2, 'value', False, 12, -3])
    # print(result)
    # result = findIntersect([1, 2, 3, 5, 7, 8, 9, 10], [1, 2, 4, 8, 9])
    # print(result)
    # result = arrange([-1, 2, -3, 5, 7, 8, 9, -10])
    # print(result)
    # result = add([1, 2, 3], [4, 5, 6])
    # print(result)
    # result = findSecond([['S ROY', 1.0], ['B BOSE', 3.0], ['N KAR', 2.0], ['C DUTTA', 2.0], ['G GHOSH', 1.0]])
    # print(result)
    # result = findPalindromes(['php', 'w3r', 'Python', 'abcd', 'Java', 'aaa'])
    # print(result)
    # result = findAnagrams(['bcda', 'abce', 'cbda', 'cbea', 'adcb'], 'abcd')
    # print(result)
    # result = findNumbers('sdf 23 safs8 5 sdfsd8 sdfs 56 21sfs 20 5')
    # print(result)
    # result = sumPN([2, 4, -6, -9, 11, -12, 14, -5, 17])
    # print(result)
    # result = removeNone([12, 0, None, 23, None, -55, 234, 89, None, 0, 6, -12])
    # print(result)
    # result = removeWords(['orange', 'red', 'green', 'blue', 'white', 'black'], ['orange', 'black'])
    # print(result)
    # result = countOcc([3, 4, 5, 8, 0, 3, 8, 5, 0, 3, 1, 5, 2, 3, 4, 2])
    # print(result)
    # result = sortStrs(['4', '12', '45', '7', '0', '100', '200', '-12', '-500'])
    # print(result)
    # result = findAvgs(((10, 10, 10), (30, 45, 56), (81, 80, 39), (1, 2, 3)))
    # print(result)
    # result = findMul([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(result)
    # result = reverseStrs(['Red', 'Green', 'Blue', 'White', 'Black'])
    # print(result)
    # result = find3('lkseropewdssafsdfafkpwe')
    # print(result)
    # result = mergeDict({'foo': 'bar'}, {'new': 'old'})
    # print(result)
    # result = mergeList([0, 2, 3], [1, 4])
    # print(result)
    # res0, res1 = lis([0, 2, -3, -7 , 7, 9, 12])
    # print(res0, res1)
    # pattern = '(?P<key>\s*\w+)\,\s*'
    # str = 'adafd_cdf, \n RT_STRING, \n Unsigned Int,\n'
    # ans = reg(pattern, str)
    # print(ans)
    
    # data = read_file('RunTimeVals.txt')
    # # pattern = r"^\{\n*\s*\{[\"\w]+\,\s*(?P<KEY>\"\w+\")\,\s*(?P<VALUE>\"\w+\")[\,\w\s\"]+\}\,[\w\W]+\;$"
    # pattern = r"\{?\"?(?P<key>\w+)\"[\s\,]*(?P<value>\w+)\s*\,(?P<value2>[\{\.\w\s\=\}\'\(\)\*]*)\s*\,[\w\"\/]*\s*\,\s*(?P<writable>\w+)\,?\}?"
    # pattern2(pattern, data)
=======

import re

def startFunc(first, second):
    print(f"Hello, {first}, i'm {second}")

def sfr(jobs, index):
    from heapq import heapify, heappush, heappop
    heap = []
    heapify(heap)
    for job in jobs:
        heappush(heap, job)

    sum  = 0   
    for _ in range(len(jobs)):
        item = heappop(heap)
        if item > jobs[index]:
            break
        sum += item
    return sum

def findMax(*args):
    return max(args)

def findSum(*args):
    return sum(args)

def reverseString(str):
    return str[::-1]

def findFactorial(num):
    if num <=  1:
        return 1
    return num * findFactorial(num -1)

def findLowerUpper(str):
   l = list(filter(lambda x: x>='a' and x <='z', str))
   u = list(filter(lambda x: x>='A' and x <='Z', str))
   return len(l), len(u)

def findMaxMin(l):
    return list(filter(lambda x: x % 2 == 1, l))

def findUnique(l):
    return set(l)

def findPalindrome(str):
    for i in range(len(str)):
        if str[i] != str[-1-i]:
            return False
    return True

def sortWords(str):
    return '-'.join(sorted(str.split('-')))

def findValues(n):
    return list(map((lambda x: x**2), range(1, n+1)))

def sortTubles(lst):
    return sorted(lst, key=(lambda x: x[1]))

def sortDict(lst):
    return sorted(lst, key=(lambda x: x['model']), reverse=True)

def filterLst(lst):
    funcs = [(lambda x: x % 2 == 0), (lambda x: x % 2 == 1)]
    for func in funcs:
        print(list(filter(func, lst)))
    funcs = [(lambda x: x ** 2), (lambda x: x ** 3)]
    for func in funcs:
        print(list(map(func, lst)))

def filterData(str):
    s = str.split(' ')
    return s[0].split('-') , s[-1]

def checkNum(lst):
    return list(filter(lambda x: isinstance(x, int) and (x != True and x != False), lst))

def findIntersect(lst1, lst2):
    lst = []
    for element in lst1:
        if element in lst2:
            lst.append(element)
    return lst

def arrange(lst):
    lst = sorted(lst)
    n = list(filter((lambda x: x < 0), lst))
    p = list(filter((lambda x: x >= 0), lst))
    return p + n

def add(lst1, lst2):
    pass

def findSecond(lst):
    s = sorted(set(map((lambda x: x[1]), lst)))
    m = list(filter(lambda x: x[1] == s[1], lst))
    return sorted(m, key=(lambda x: x[0]))

def findPalindromes(lst):
    def palindrome(str):
        for i in range(len(str)):
            if str[i] != str[-1-i]:
                return False
        return True
    return list(filter(lambda x: palindrome(x),lst)) 

def findAnagrams(lst, str):
    s = set(str)
    f = list(filter((lambda x: set(x) == s), lst))
    return f

def findNumbers(str):
    s = str.split(' ')
    l = []
    for number in s:
        try:
            tmp = int(number)
            l.append(tmp)
        except ValueError as e:
            pass
    return sorted(l)

def sumPN(lst):
    l = list(filter(lambda x: x<0, lst))
    p = list(filter(lambda x: x >=0, lst))
    return sum(l), sum(p)

def removeNone(lst):
    return list(filter((lambda x: x != None),lst))

def removeWords(lst, remove):
    l = []
    for num in lst:
        for rem in remove:
            if num == rem:
                break
        else:
            l.append(num)
    return l

def countOcc(lst):
    from collections import Counter
    return Counter(lst)

def sortStrs(lst):
    return sorted(lst, key=(lambda x: int(x)))

def findAvgs(lst):
    return list(map(lambda x: round(sum(x)/len(x), 2), lst))

def findMul(lst):
    from functools import reduce
    return reduce((lambda x, y: x * y), lst)

def reverseStrs(lst):
    return list(map((lambda x: x[::-1]),lst))

def find3(str):
    from collections import Counter
    c = Counter(str)
    return sorted(c.items(), key=(lambda item: item[1]), reverse=True)[0:3]

def mergeDict(dic1, dic2):
    return {**dic1, **dic2}

def mergeList(lst1, lst2):
    return [*lst1] + [*lst2]


def lis(lst):
    dp = [0] * len(lst)
    for i in range(len(lst)-1):
        for j in range(i+1, len(lst)):
            if lst[j] > lst[i] and dp[i] <= dp[j]:
                dp[j] = dp[i] + 1    
    
    print(dp)
    maxIndex = lst[dp.index(max(dp))]
    leastIndex = lst[dp.index(max(dp)) - max(dp)]
    return leastIndex, maxIndex

def reg(pattern, str):
    import re
    s = str.split('\n')
    res = re.search(pattern, str, re.MULTILINE)
    res = res.group('key')
    return res


def read_file(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def pattern2(pattern, str):
    result = re.findall(pattern, str)
    print(result)
    pass

if  __name__ == '__main__':
    #startFunc('Mina', 'Shenouda')
    # result = sfr([3, 10, 20, 1, 2], 0)
    # print(result)
    # result = sfr([2, 10, 10, 1, 3], 2)
    # print(result)
    # result = sfr([10, 10, 10, 10], 3)
    # print(result)
    # result = findMax(1, 2 , 3)
    # print(result)
    # result = findSum(1, 3, 6 , 8 , 9)
    # print(result)
    # result = reverseString('minaShenouda')
    # print(result)
    # result = findFactorial(5)
    # print(result)
    # result, result1 = findLowerUpper('The quick Brow Fox')
    # print(result, result1)
    # # result = findMaxMin([1, 5, 9, 12])
    # print(result)
    # result = findUnique([1, 2, 3, 3, 3 , 3 , 3])
    # print(result)
    # result = findPalindrome('cannac')
    # print(result)
    # result = sortWords('green-red-yellow-black-white')
    # print(result)
    # result = findValues(30)
    # print(result)
    # result = (lambda x: x + 15)
    # print(result(5))
    # result = (lambda x, y: x * y)
    # print(result(5, 15))
    # result = (lambda x: 15 * x)
    # print(result(4))
    # result = sortTubles([('English', 88), ('Science', 90), ('Maths', 97), ('Social sciences', 82)])
    # print(result)
    # result = sortDict([{'make': 'Nokia', 'model': 216, 'color': 'Black'}, {'make': 'Samsung', 'model': 7, 'color': 'Blue'}, {'make': 'Mi Max', 'model': 2, 'color': 'Gold'}])
    # print(result)
    # result = filterLst([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(result)
    # result = filterData('2020-01-15 09:03:32.744178')
    # print(result)
    # result = checkNum([2, 'value', False, 12, -3])
    # print(result)
    # result = findIntersect([1, 2, 3, 5, 7, 8, 9, 10], [1, 2, 4, 8, 9])
    # print(result)
    # result = arrange([-1, 2, -3, 5, 7, 8, 9, -10])
    # print(result)
    # result = add([1, 2, 3], [4, 5, 6])
    # print(result)
    # result = findSecond([['S ROY', 1.0], ['B BOSE', 3.0], ['N KAR', 2.0], ['C DUTTA', 2.0], ['G GHOSH', 1.0]])
    # print(result)
    # result = findPalindromes(['php', 'w3r', 'Python', 'abcd', 'Java', 'aaa'])
    # print(result)
    # result = findAnagrams(['bcda', 'abce', 'cbda', 'cbea', 'adcb'], 'abcd')
    # print(result)
    # result = findNumbers('sdf 23 safs8 5 sdfsd8 sdfs 56 21sfs 20 5')
    # print(result)
    # result = sumPN([2, 4, -6, -9, 11, -12, 14, -5, 17])
    # print(result)
    # result = removeNone([12, 0, None, 23, None, -55, 234, 89, None, 0, 6, -12])
    # print(result)
    # result = removeWords(['orange', 'red', 'green', 'blue', 'white', 'black'], ['orange', 'black'])
    # print(result)
    # result = countOcc([3, 4, 5, 8, 0, 3, 8, 5, 0, 3, 1, 5, 2, 3, 4, 2])
    # print(result)
    # result = sortStrs(['4', '12', '45', '7', '0', '100', '200', '-12', '-500'])
    # print(result)
    # result = findAvgs(((10, 10, 10), (30, 45, 56), (81, 80, 39), (1, 2, 3)))
    # print(result)
    # result = findMul([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(result)
    # result = reverseStrs(['Red', 'Green', 'Blue', 'White', 'Black'])
    # print(result)
    # result = find3('lkseropewdssafsdfafkpwe')
    # print(result)
    # result = mergeDict({'foo': 'bar'}, {'new': 'old'})
    # print(result)
    # result = mergeList([0, 2, 3], [1, 4])
    # print(result)
    # res0, res1 = lis([0, 2, -3, -7 , 7, 9, 12])
    # print(res0, res1)
    # pattern = '(?P<key>\s*\w+)\,\s*'
    # str = 'adafd_cdf, \n RT_STRING, \n Unsigned Int,\n'
    # ans = reg(pattern, str)
    # print(ans)
    
    # data = read_file('RunTimeVals.txt')
    # # pattern = r"^\{\n*\s*\{[\"\w]+\,\s*(?P<KEY>\"\w+\")\,\s*(?P<VALUE>\"\w+\")[\,\w\s\"]+\}\,[\w\W]+\;$"
    # pattern = r"\{?\"?(?P<key>\w+)\"[\s\,]*(?P<value>\w+)\s*\,(?P<value2>[\{\.\w\s\=\}\'\(\)\*]*)\s*\,[\w\"\/]*\s*\,\s*(?P<writable>\w+)\,?\}?"
    # pattern2(pattern, data)
>>>>>>> 6491f59943a7f2d401bac25b7dedf78025fc6ea7
    pass