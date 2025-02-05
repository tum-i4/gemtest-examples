"""
Implementation source from: 
https://github.com/anxiaonong/Maxflow-Algorithms/blob/master/Dinic's%20Algorithm.py
"""

# build level graph by using BFS
def Bfs(C, F, s, t):  # C is the capacity matrix
    n = len(C)
    queue = []
    queue.append(s)
    global level
    level = n * [0]  # initialization
    level[s] = 1
    while queue:
        k = queue.pop(0)
        for i in range(n):
            if (F[k][i] < C[k][i]) and (level[i] == 0):  # not visited
                level[i] = level[k] + 1
                queue.append(i)
    return level[t] > 0


# search augmenting path by using DFS
def Dfs(C, F, k, cp):
    tmp = cp
    if k == len(C) - 1:
        return cp
    for i in range(len(C)):
        if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
            f = Dfs(C, F, i, min(tmp, C[k][i] - F[k][i]))
            F[k][i] = F[k][i] + f
            F[i][k] = F[i][k] - f
            tmp = tmp - f
    return cp - tmp


# calculate max flow
# _ = float('inf')
def MaxFlow(C, s, t):
    n = len(C)
    F = [n * [0] for i in range(n)]  # F is the flow matrix
    flow = 0
    while Bfs(C, F, s, t):
        flow = flow + Dfs(C, F, s, 100000)
    return flow