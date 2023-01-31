from cmath import inf
from math import ceil
from collections import deque, defaultdict, Counter
from heapq import heapify, heappush, heappop
from multiprocessing import heap, queues
from re import T
from tkinter import N
from copy import deepcopy
from dis import dis
from functools import reduce
from inspect import _void
import os
import sys
import collections
from pip import List

def networkDelayTime(times, n: int, k: int) -> int:
    graph = defaultdict(list)
    for src, dst, c in times:
        graph[src].append((dst, c)) 
    
    
    queue = [(0, k)] #(cost, node)
    visited = set()
    max_cost = 0    
    while queue:
        cost, node = heappop(queue)
        if node in visited:
            continue
            
        visited.add(node)
        max_cost = max(max_cost, cost)
        neighbours = graph[node]       
        for neighbour in neighbours:
            new_node, new_cost = neighbour
            if new_node not in visited:               
                curr_cost =  cost + new_cost
                heappush(queue, (curr_cost, new_node))
        
    # print(visited)
    return max_cost if len(visited) == n else -1

def adjMatrix(n ,edges):
    matrix = [[0]*n for _ in range(n)]
    for edge in edges:
        i, j = edge
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix

def restoreArray(adjacentPairs):
    adj, ans, seen, start = defaultdict(list), [], set(), 0
    # DFS Helper for graph traversal
    def dfs(i):
        ans.append(i)
        seen.add(i)
        for nei in adj[i]:
            if nei not in seen: 
                dfs(nei)
    # Create Adjacency List
    for a,b in adjacentPairs:
        adj[a].append(b)
        adj[b].append(a)
    print(adj)
    # Find the degree 1 node - starting point for DFS
    for x in adj:
        if len(adj[x]) == 1:
            start = x        
    # Call DFS to populate path/ans
    dfs(start)
    return ans

#adjacentPairs = [[2,1],[3,4],[3,2]]
# adjacentPairs = [[4,-2],[1,4],[-3,1]]
# result = restoreArray(adjacentPairs)
    
# times = [[2,1,1],[2,3,1],[3,4,1]]
# n = 4
# k = 2
# # times = [[1,2,1]]
# # n = 2
# # k = 1
# times = [[1,2,1]]
# n = 2
# k = 2
# # times = [[3,5,78],[2,1,1],[1,3,0],[4,3,59],[5,3,85],[5,2,22],[2,4,23],[1,4,43],[4,5,75],[5,1,15],[1,5,91],[4,1,16],[3,2,98],[3,4,22],[5,4,31],[1,2,0],[2,5,4],[4,2,51],[3,1,36],[2,3,59]]
# # n = 5
# # k = 5
# result = networkDelayTime(times, n, k)
# print('The result is {}'.format(result))

# adjacentPairs = [[2,1],[3,4],[3,2]]
# result = restoreArray(adjacentPairs)
# print('The result is {}'.format(result))

def isBipartite(graph) -> bool:
    length = len(graph)
    visited = length * [False] 
    marked =  length * [False]

    def dfs(cur):
        visited[cur] = True
        for neighbor in graph[cur]:
            if not visited[neighbor]:
                marked[neighbor] = not marked[cur]
                if not dfs(neighbor):
                    return False
            else:
                if marked[neighbor] == marked[cur]:
                    return False
        return True

    for i in range(length):
        if not visited[i]:
            if not dfs(i):
                return False
    return True

def getAncestors(n: int, edges) -> list[list[int]]:
    reversed = []
    for i, edge in enumerate(edges):
        reversed += [edge[::-1]]
    
    del edges
    adjList = {}    
    for i, edge in enumerate(reversed):
        u, v = edge
        if not adjList.get(u):
            adjList[u] = [v]
        else:
            adjList[u] += [v] 
    
    print(adjList) 
    visited = (n+1) * [False]   
    ansectors = {}
    
    def dfs(u): 
        visited[u] = True
        if adjList.get(u): 
            return adjList[u] 
        for v in adjList[u]:
            ansectors[u] += dfs(v)
                    
    for i in range(n):
        if not visited[i]:
            if not adjList.get(i):
                adjList[i] = []
                visited[i] = True
            else: 
                dfs(i) 
           
def adjMatrix(n ,edges):
    matrix = [[0]*n for _ in range(n)]
    for edge in edges:
        i, j = edge
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix

def adjMatrix2(n, edges, map):  
    matrix = [[0]*n for _ in range(n)]
    for edge in edges:
        u, v = edge
        i, j = map[u], map[v]
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix

def adjList(n, edges):        
    obj = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for j in edge:
            obj[i] +=[j]   
    return obj

def fillListObj(n, edges):
    obj = {}
    i = 0
    for j, edge in enumerate(edges):
        x, y = edge
        if not obj.__contains__(x):
            obj[x] = i
            i += 1
        if not obj.__contains__(y):
            obj[y] = i
            i += 1   
    return obj

def traverseDfs(n, edges, map):
    adj = adjMatrix2(n, edges, map)
    visited = n*[False]
    def dfs(i):
        result = []
        if adj[i] != []:
            result = [i]
        visited[i] = True
        for neighbor in adj[i]:
            if not visited[neighbor]:
                result += dfs(neighbor) 
        return result
   
    result = [] 
    for i in range(n):
        if not visited[i]:
            result += dfs(i)
    return result

def traverseBfs(n, edges):
    adj = adjList(n, edges)
    visited = n*[False]
    queue = []
    
    def bfs(i):
        queue.append(i)
        visited[i] = True
        result = []
        while len(queue) > 0:
            root = queue.pop(0)
            result += [root]
            for neighbor in adj[root]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        return result
    
    result = []
    for i in range(n):
        if not visited[i]:
            result = bfs(i)
    return result

def detectCylce(n, edges):
    adj = adjList(n, edges)
    visited = n*[False]
    
    def dfs(i):
        if visited[i] == True:
            return True

        visited[i] = True
        result = False
        for neighbor in adj[i]:
            result = dfs(neighbor) 
        return result
   
    result = False
    for i in range(n):
        if not visited[i]:
            result = dfs(i)
    return result
    

def detectCycleUnDirected(n, edges):
    adj = adjList(n, edges)
    visited = n*[0]
    
    def dfs(i):
        visited[i] += 1
        if visited[i] == 2:
            return True
        
        result = False
        for neighbor in adj[i]:
            result = dfs(neighbor) 
        return result
   
    result = False
    for i in range(n):
        if visited[i] == 0:
            result = dfs(i)
    return result

def topologicalSort(n, edges):
    adj = adjList(n, edges)
    visited = n*[False]
    
    stack = [] 
    def dfs(i):
        visited[i] = True
        for neighbor in adj[i]:
            if not visited[neighbor]:
                dfs(neighbor) 
                stack.append(neighbor)
   
    for i in range(n):
        if not visited[i]:
            dfs(i)
            stack.append(i)
    
    result = []
    while len(stack) > 0:
        result.append(stack.pop(0))
    return result
    
def topologicalSortKahn(n, edges):
    adj = adjList(n, edges)
            
    indegree = [0] * n
    for edge in edges:
        _, t = edge
        indegree[t] += 1
 
    queue = []
    for i in range(len(indegree)):
        if indegree[i] == 0:
            queue.append(i)

    count = 0
    result = []
    while len(queue) > 0:
        i = queue.pop(0)
        result.append(i)
        count+= 1
             
        for neighbor in adj[i]:
            if indegree[neighbor] > 0:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
    
    result = result[::-1]
    return result
    if count == n:
        return True
    return False

def spanningTree(graph): 
    import math
    V = len(graph)
    values = [math.inf] * V
    parent = [0] * V
    mst = [False] * V
    
    parent[0] = -1
    values[0] = 0

    def findMinimum():
        minimum = math.inf
        vertex = V
        for i in range(V):
            if mst[i] == False and values[i] < minimum:
                vertex = i
                minimum = values[i]
        return vertex   
    
    def dfs():
        for _ in range(V-1):
            U = findMinimum()
            mst[U] = True
            for j in range(V):
                if graph[U][j] != 0 and mst[j] == False and graph[U][j] < values[j]:
                    values[j] = graph[U][j]
                    parent[j] = U
    
    dfs()
    return values, parent

def minCostConnectPoints(points) -> int:
    n = len(points)
    graph = [[0]*n for _ in range(n)]
    dist = 0 
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                graph[i][j] = dist
                graph[j][i] = dist
    
    
    cost, parent = spanningTree(graph)
    total = reduce(lambda x,y: x + y, cost)
    print(cost, parent, total)
   

def canFinish(numCourses: int, prerequisites) -> bool:
    n = numCourses
    edges = prerequisites
    result = topologicalSortKahn(n, edges)
    return result    
      

def findOrder(numCourses: int, prerequisites) -> List[int]:
    n = numCourses
    edges = prerequisites
    result = topologicalSortKahn(n, edges)
    return result

def traverseBfs(n, edges):
    adj = adjList(n, edges)
    visited = n*[False]
    queue = []
    
    def bfs(i):
        queue.append(i)
        visited[i] = True
        result = []
        while len(queue) > 0:
            root = queue.pop(0)
            result += [root]
            for neighbor in adj[root]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
        return result
    
    result = []
    for i in range(n):
        if not visited[i]:
            result = bfs(i)
    return result


def allPathsSourceTarget(graph: List[List[int]]) -> List[List[int]]:
    n = len(graph)
    from copy import deepcopy
  
    result = []
    def dfs(i, tmp):
        if i == n-1: return True 
        if len(graph[i]) == 0: return False
        
        flag = False
        for j in graph[i]:
            tmp.append(j)
            flag = dfs(j, tmp)
            if flag:
                result.append(deepcopy(tmp))
            tmp.pop()                 
            flag = False
        return False
    
    dfs(0, [0])
    return result 

#graph = [[1,2],[3],[3],[]]
# graph = [[4,3,1],[3,2,4],[3],[4],[]]
#graph = [[4,3,1],[3,2,4],[],[4],[]]
# result = allPathsSourceTarget(graph)    
# print(f'The result is {result}')

def findSmallestSetOfVertices(n: int, edges: List[List[int]]) -> List[int]:
    fr = set()
    to = set()
    for i, edge in enumerate(edges):
        x, y = edge
        fr.add(x)
        to.add(y)

    result = []
    for key in fr:
        if not key in to:
            result.append(key)

    print(fr, to)
    return result
# n = 6
# edges = [[0,1],[0,2],[2,5],[3,4],[4,2]]
# n = 5
# edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
# result = findSmallestSetOfVertices(n, edges)
# print(f'The result is {result}')

def findRedundantConnection(edges: List[List[int]]) -> List[int]:
    graph = collections.defaultdict(set)
    def dfs(source, target):
        if source not in seen:
            seen.add(source)
            if source == target: return True
            return any(dfs(nei, target) for nei in graph[source])

    for u, v in edges:
        seen = set()
        if u in graph and v in graph and dfs(u, v):
            return u, v
        graph[u].add(v)
        graph[v].add(u)
    
    print(graph, seen)

#edges = [[1,2],[1,3],[2,3]]
#edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
# edges = [[9,10],[5,8],[2,6],[1,5],[3,8],[4,9],[8,10],[4,10],[6,8],[7,9]]
# result = findRedundantConnection(edges)
# print(f'The result is {result}')

def minReorder(n: int, connections: List[List[int]]) -> int:
    graph = defaultdict(set)
    for a, b in connections:
        graph[a].add((b, True))
        graph[b].add((a, False))
    
    queue = deque([(0, False)])
    ans = 0
    visited = set()
    while queue:
        city, needs_flipped = queue.popleft()
        visited.add(city)
        if needs_flipped:
            ans += 1
        for neighbour in graph[city]:
            if neighbour[0] not in visited:
                queue.append(neighbour)
    return ans
   
# n = 5
# connections = [[1,0],[1,2],[3,2],[3,4]]
# result = minReorder(n, connections)
# print(f'The result is {result}')

def makeConnected(n: int, connections: List[List[int]]) -> int:
    rank = [0]*n
    parent = [0]*n
    
    def makeSet(n):
        for i in range(n):
            parent[i] = i
            rank[i] = 0
             
    def findParent(x): 
        if parent[x] == x: return x
        parent[x] = findParent(parent[x])
        return parent[x]
        
    def Union(x , y): 
        x = findParent(x)
        y = findParent(y)
        if(rank[x]<rank[y]):
            parent[x] = y; 
        elif(rank[x]>rank[y]):
            parent[y] = x
        else:
            parent[y] = x
            rank[x]+=1
    
    m = len(connections)
    if(m+1 < n):
        return -1;
    makeSet(n);
    for i in connections: 
        u = i[0]
        v = i[1]
        Union(u,v)
    
    result = 0;
    for i in range(n):
        if(findParent(i) == i):
            result+=1
    return result-1
 
# n = 4
# connections = [[0,1],[0,2],[1,2]]
# n = 6
# connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
# n = 6
# connections = [[0,1],[0,2],[0,3],[1,2]]
# result = makeConnected(n, connections)
# print(f'The result is {result}')
   
def countPaths(n: int, roads: List[List[int]]) -> int:
    graph = defaultdict(list)
    for src, dst, c in roads:
        graph[src].append((dst, c)) 
        graph[dst].append((src, c))    
        
    
    dist = [0]*n
    def dijkstra():
        queue = [(0, 0)] #(cost, node)
        visited = set()
        max_cost = 0   
        while queue:
            cost, node = heappop(queue)
            if node in visited:
                continue
                
            visited.add(node)
            max_cost = max(max_cost, cost)
            dist[node] = max_cost
            neighbours = graph[node]       
            for neighbour in neighbours:
                new_node, new_cost = neighbour
                if new_node not in visited:               
                    curr_cost =  cost + new_cost
                    heappush(queue, (curr_cost, new_node))
        return max_cost if len(visited) == n else -1
    
    minDist = dijkstra()

    dp = {}
    mod = 10**9 + 7
    def dfs(src):    
        if src == n-1: return 1
        if dp.get(src): return dp[src] 
       
        result = 0
        distance = dist[src]
        for dst, c in graph[src]:
            if c + distance == dist[dst]:
                result += dfs(dst) % mod
        dp[src] = result
        return dp[src]         
            
    result = dfs(0)
    print(dp)
    return result
            
# n = 7
# roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]
# n = 2
# roads = [[1,0,10]]
# result = countPaths(n, roads)
# print(f'The result is {result}')
def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    graph = defaultdict(list)
    for u, v, c in flights:
        graph[u].append((v, c)) 

    def dijkstra(src, dst):
        queue = [(src, 0, k+1)] #(src, cost, stops)
        visited = set()
        max_cost = 0  
        while queue:
            node, cost, stops = heappop(queue)
            if node == dst: return cost
            
            if node in visited:
                continue
            visited.add(node)
            if stops > 0:  
                for neighbour in graph[node]:
                    new_node, new_cost = neighbour
                    max_cost = max(max_cost, cost)
                    if new_node not in visited:               
                        curr_cost = cost + new_cost
                        stops -= 1
                        heappush(queue, (new_node, curr_cost, stops))
        return max_cost if max_cost > 0 else -1
    
    minDist = dijkstra(src, dst)
    return minDist
    
# n = 4
# flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
# src = 0
# dst = 3
# k = 1
# n = 3
# flights = [[0,1,100],[1,2,100],[0,2,500]]
# src = 0
# dst = 2
# k = 1
#n = 4
# flights = [[0,1,1],[0,2,5],[1,2,1],[2,3,1]]
# src = 0
# dst = 3
# k = 1
# result = findCheapestPrice(n, flights, src, dst, k)
# print(f'The result is {result}')

def watchedVideosByFriends(watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
    
    d = defaultdict(list)
    def bfs(id):
        k = 0
        queue = [(id, k)] #id, level
        visited = set()
        while queue:
            node, k = heappop(queue)
            if not node in visited:
                d[k] += watchedVideos[node]
            visited.add(node)
            if k < level:
                k += 1 
                for neighbor in friends[node]:
                    if not neighbor in visited:
                        heappush(queue, (neighbor, k)) 
                    
    
    bfs(id)
    s = sorted(Counter(d[level]).items(), key=lambda kv: (kv[1], kv[0]))
    result = []
    for k, v in s:
        result.append(k)
    return result
    
# watchedVideos = [["A","B"],["C"],["B","C"],["D"]]
# friends = [[1,2],[0,3],[0,3],[1,2]]
# id = 0
# level = 2
# watchedVideos = [["xk","qvgjjsp","sbphxzm"],["rwyvxl","ov"]]
# friends = [[1],[0]]
# id = 0
# level = 1
# result = watchedVideosByFriends(watchedVideos, friends, id, level)
# print(f'The result is {result}')
def edgeScore(edges: List[int]) -> int:
    graph = defaultdict(list)
    for i, edge in enumerate(edges):
        graph[edge] += [i] 
    
    index = float("inf")
    score = 0
    maxVal = 0
    for k, v in graph.items():
        score = sum(v)
        if score > maxVal:
            maxVal = score
            index = k
        elif score == maxVal:
            index = min(index, k)

    print(graph)    
    return index


# edges = [1,0,0,0,0,7,7,5]
# edges = [2,0,0,2]
# result = edgeScore(edges)
# print(f'The result is {result}')

def gardenNoAdj(n: int, paths: List[List[int]]) -> List[int]:
    adj, q, visited = defaultdict(list), deque(), set()
    # Make adj List
    for u,v in paths:
        adj[u-1].append(v-1)
        adj[v-1].append(u-1)
    
    # Graph coloring for all connected components
    color = [0]*n
    for u in range(n):
        if u not in visited:
            q.append(u)
            color[u] = 1
            while q:
                node = q.popleft()
                visited.add(node)
                for nei in adj[node]:
                    if color[nei] == 0 or color[nei] == color[node]:
                        color[nei] = (color[node] + 1) % 4 if (color[node] + 1) > 4 else (color[node] + 1)
                        q.append(nei)
    return color

# n = 3
# paths = [[1,2],[2,3],[3,1]]
# n = 4
# paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
# n = 4
# paths = [[1,2],[3,4]]
# result = gardenNoAdj(n, paths)
# print(f'The result is {result}')

def reachableNodes(n: int, edges: List[List[int]], restricted: List[int]) -> int:
    graph, visited = defaultdict(list), set()
    for u, v in edges:
        graph[u] += [v]
        graph[v] += [u]
    
    
    for i in restricted:
        visited.add(i)
    
    result = 0
    def dfs(i):
        visited.add(i)
        
        result = 1 
        for neighbor in graph[i]:
            if not neighbor in visited:
                result += dfs(neighbor)
        return result
    result = dfs(0)
    return result     
   
# n = 7
# edges = [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]]
# restricted = [4,5]
# n = 7
# edges = [[0,1],[0,2],[0,5],[0,4],[3,2],[6,5]]
# restricted = [4,2,1]
# result = reachableNodes(n, edges, restricted)
# print(f'The result is {result}')
def checkIfPrerequisite(numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
    
    graph = defaultdict(list)
    for (u, v) in prerequisites:
        graph[v] += [u]
    
    def bfs(i, visited):
        if i == u: return True
        if i in visited: return False
        visited.add(i)
        for neighbor in graph[i]:
            if not neighbor in visited:
                if bfs(neighbor, visited): return True
        return False
        
    result = [False] * len(queries)
    for i, (u,v) in enumerate(queries):
        visited = set()
        result[i] = bfs(v, visited)    
    return result
    
# numCourses = 2
# prerequisites = [[1,0]]
# queries = [[0,1],[1,0]]

# numCourses = 3
# prerequisites = [[1,2],[1,0],[2,0]]
# queries = [[1,0],[1,2]]
# numCourses = 2
# prerequisites = []
# queries = [[1,0],[0,1]]

# numCourses = 4
# prerequisites = [[2,3],[2,1],[0,3],[0,1]]
# queries = [[0,1],[0,3],[2,3],[3,0],[2,0],[0,2]]
# result = checkIfPrerequisite(numCourses, prerequisites, queries)
# print(f'The result is {result}')

def countPairs(n: int, edges: List[List[int]]) -> int: 
    def unionFind(n,edges):   
        rank = [1]*n
        parent = [0]*n
        
        parent = [i for i in range(n)]
                       
        def Find(x):
            res = x
            while res != parent[res]:
                # parent[res] = parent[parent[res]]
                res = parent[res] 
            return res
        
        def Union(x , y): 
            p1, p2 = Find(x), Find(y)
            if p1 == p2: return 0
            if rank[p2] > rank[p1]:
                parent[p1] = p2
                rank[p2] += rank[p1]
            else:
                parent[p2] = p1
                rank[p1] += rank[p2]
            return 1
        
        for edge in edges:
            #Union(edge[0], edge[1])
            Union(edge[1], edge[0])
        return rank, parent
    rank, parent = unionFind(n, edges)
    print(rank, parent)
    result = 0
    # m = len(parent)
    # for i, x in enumerate(parent):
    #     for j in range(i+1, m):
    #         if x != parent[j]:
    #             result += 1
 
    # return result

# n = 7
# edges = [[0,2],[0,5],[2,4],[1,6],[5,4]]
# n = 3
# edges = [[0,1],[0,2],[1,2]]
# n = 11
# edges = [[5,0],[1,0],[10,7],[9,8],[7,2],[1,3],[0,2],[8,5],[4,6],[4,2]]
# result = countPairs(n, edges)
# print(f'The result is {result}')
def findTheCity(n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    adj = defaultdict(set)
    for src, dst, w in edges:
        adj[src].add((dst,w))
        adj[dst].add((src,w))
    
    def shortestD(cur):
        dst = [float('inf') for _ in range(n)]
        dst[cur] = 0
        heap = [(0, cur)]
        visited = set()
        while heap:
            d, i = heappop(heap)
            
            visited.add(i)
        
            for nei, w in adj[i]:
                if nei not in visited and dst[nei] > dst[i]+w:
                    dst[nei] = dst[i]+w
                    heappush(heap, (dst[nei], nei))
        
        print(dst)
        ans = 0
        for i, d in enumerate(dst):
            if i != cur and d <= distanceThreshold:
                ans += 1
        
        return ans
    
    minC = float('inf')
    ans = -1
    for i in range(n):
        c = shortestD(i)
        print(c)
        if c <= minC:
            minC = c
            ans = i
    
    return ans

# n = 4
# edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]]
# distanceThreshold = 4

# n = 5
# edges = [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]]
# distanceThreshold = 2

# n = 34
# edges = [[3,8,6065],[27,33,6596],[1,21,5037],[24,27,7612],[2,12,9802],[0,22,5578],[7,30,8719],[4,9,8316],[9,29,2750],[13,18,477],[32,33,2431],[19,22,4099],[4,15,3624],[8,26,9221],[17,32,2186],[9,24,1848],[2,16,3025],[27,30,6736],[11,12,821],[7,10,1626],[0,30,8941],[1,8,4354],[2,32,1753],[17,26,3348],[23,27,4288],[8,23,1095],[21,22,9359],[15,18,8625],[18,24,1287],[2,31,1193],[13,15,3562],[5,8,2841],[4,22,8381],[16,18,7080],[16,33,358],[1,14,9673],[28,29,6032],[8,31,7974],[23,28,4649],[16,29,3604],[1,5,3284],[9,15,9799],[20,29,8088],[8,15,3854],[6,25,6971],[9,31,7409],[12,13,6016],[13,24,8921],[4,33,3094],[2,14,7900],[10,21,1192],[4,10,4204],[19,23,6674],[6,14,3300],[24,29,136],[20,24,8717],[19,27,6238],[5,27,8427],[25,28,7981],[9,17,1252],[1,15,6615],[10,27,8357],[2,18,9475],[2,33,9579],[4,26,6973],[0,14,658],[22,23,5765],[6,11,7512],[3,19,105],[12,19,3110],[1,11,4905],[3,28,91],[4,28,8861],[10,30,1967],[0,32,4959],[5,18,8397],[3,15,5171],[14,15,8897],[15,27,9372],[4,32,9034],[9,14,4629],[4,25,8612],[27,29,6741],[4,29,8881],[6,13,8485],[6,10,6690],[10,13,9876],[7,31,9521],[8,33,5043],[24,30,7415],[0,33,4947],[7,27,2146],[13,21,8296],[2,5,7278],[5,15,9606],[15,21,2300],[5,11,9012],[5,22,2671],[13,25,4141],[3,20,158],[24,25,6950],[7,15,9272],[0,5,594],[4,8,6036],[0,17,6896],[3,24,6589],[10,15,4613],[17,23,301],[8,18,1483],[18,19,1476],[31,33,79],[5,26,6282],[23,29,4406],[7,9,7609],[10,24,4456],[17,24,6106],[8,13,7888],[3,27,5514],[6,18,6365],[25,26,7474],[0,27,1909],[3,25,7926],[8,14,5809],[0,20,2371],[17,28,6803],[20,23,2430],[0,23,298],[22,29,2820],[0,19,4264],[15,25,6026],[8,27,2083],[22,24,9660],[1,20,4705],[29,32,6766],[2,28,4226],[11,18,8418],[20,21,4707],[3,17,6894],[2,27,4484],[7,17,7103],[5,12,5504],[25,30,7960],[18,23,3531],[13,26,8051],[2,6,6585],[6,22,6966],[21,33,1498],[3,22,1056],[28,32,2122],[2,9,3378],[16,27,2452],[6,21,3756],[23,31,429],[1,17,1692],[11,30,4149],[3,18,2552],[25,29,7861],[16,26,1622],[11,20,6540],[9,11,3071],[13,20,50],[12,18,1461],[21,31,7008],[0,10,834],[21,27,9005],[12,16,3577],[22,31,8758],[30,31,6913],[18,22,5681],[1,2,771],[2,3,1691],[9,21,9058],[4,23,6876],[8,16,1944],[10,16,4348],[10,29,4568],[14,29,2934],[12,27,7860],[16,30,782],[0,8,3510],[10,26,1429],[16,20,6386],[1,9,2029],[20,25,7329],[11,27,4821],[28,31,8321],[20,33,8159],[3,6,6441],[2,19,1904],[3,30,9931],[13,28,7852],[0,4,1734],[23,30,2444],[0,21,6331],[6,26,3297],[0,29,2739],[17,22,8532],[9,23,4221],[26,27,6826],[2,4,8794],[6,7,4729],[7,11,8069],[1,23,8926],[3,7,3517],[13,14,5523],[4,19,7963],[10,14,1686],[2,10,141],[17,27,5684],[18,25,4384],[1,12,925],[8,11,8857],[3,4,7214],[3,23,4913],[7,32,1651],[16,25,3745],[19,28,8324],[1,25,4499],[0,25,4430],[1,13,4037],[5,28,6745],[19,20,2431],[0,31,2134],[9,12,4200],[7,12,3200],[26,32,6681],[14,17,9189],[29,30,9806],[8,28,958],[10,23,8730],[6,9,9978],[12,31,8346],[12,20,2439],[25,33,6780],[22,26,4427],[0,9,4585],[5,25,7867],[18,30,5011],[6,16,4376],[13,29,8050],[12,22,3513],[15,23,8172],[13,23,6025],[0,15,9815],[0,12,7710],[11,16,3960],[31,32,5545],[10,20,2887],[8,10,9925],[13,17,2969],[11,17,9512],[13,31,7392],[1,27,8762],[0,28,2449],[0,18,953],[14,19,8257],[19,33,5342],[1,28,8659],[3,31,2213],[11,15,3493],[5,9,5167],[15,33,8090],[7,23,7871],[14,28,5408],[2,8,1940],[23,32,2096],[7,33,2296],[4,13,4202],[19,30,3687],[7,25,1443],[11,19,8829],[12,24,820],[20,31,9226],[14,20,2820],[21,24,1903],[23,25,3707],[5,13,9229],[6,30,3268],[26,31,8242],[3,33,9300],[9,32,5045],[3,21,6919],[24,31,5369],[15,20,70],[8,20,329],[19,32,5003],[15,28,3609],[6,24,1386],[3,26,3679],[18,31,4591],[19,24,5589],[9,33,4409],[4,31,9850],[11,33,8494],[0,26,6215],[15,16,379],[17,21,1994],[11,32,5405],[6,12,5686],[9,16,2285],[16,32,1858],[30,33,4110],[4,16,2348],[5,21,9405],[3,29,673],[14,23,5686],[16,28,1268],[18,21,1505],[12,17,1691],[12,23,4915],[4,20,5195],[6,29,4079],[1,16,4413],[2,20,8678],[8,32,816],[22,33,5928],[15,24,511],[16,17,1284],[24,33,2278],[5,32,6543],[1,4,6096],[7,14,3966],[10,28,1538],[1,19,5388],[13,16,4484],[12,26,131],[0,24,8442],[17,25,5273],[8,12,1839],[18,29,5774],[8,21,2063],[4,11,9932],[26,33,4442],[2,15,6639],[5,6,1493],[9,27,9448],[7,8,8647],[4,14,7792],[5,29,9248],[0,6,6861],[11,13,8778],[1,6,6452],[2,29,4934],[4,17,3595],[26,28,4959],[11,28,8997],[2,17,2182],[12,33,884],[27,31,9832],[20,27,8332],[11,26,1801],[4,27,2870],[17,18,3942],[11,31,3523],[26,29,7121],[15,22,4498],[1,3,8945],[19,25,328],[22,28,4103],[5,23,8829],[6,31,4439],[7,16,8686],[20,28,4289],[6,23,7754],[12,30,2066],[20,22,6608],[9,18,1700],[6,8,6120],[14,25,1132],[9,20,8917],[12,25,5950],[11,21,8926],[15,32,9102],[26,30,8313],[13,22,9517],[15,30,499],[13,27,5049],[22,25,7299],[9,13,2167],[21,32,2553],[8,9,1219],[3,9,9491],[24,28,2326],[14,16,3544],[14,22,7932],[13,32,5497],[27,28,5982],[11,29,4790],[21,25,2618],[0,2,2550],[10,11,6255],[18,32,7205],[6,19,6647],[21,23,1932],[12,14,9847],[1,26,2379],[8,25,4420],[18,20,4839],[19,21,9891],[14,18,135],[15,26,8803],[5,24,159],[6,28,2173],[9,25,6218]]
# distanceThreshold = 9207

# n = 5
# edges = [[0,1,2],[0,4,8],[1,2,10000],[1,4,2],[2,3,10000],[3,4,1]]
# distanceThreshold = 10000

n = 6
edges = [[0,1,10],[0,2,1],[2,3,1],[1,3,1],[1,4,1],[4,5,10]]
distanceThreshold = 20

result = findTheCity(n, edges, distanceThreshold)  
print(f'The result is {result}')
