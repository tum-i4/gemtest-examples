# Source: https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/

# A Dynamic Programming based Python
# Program for 0-1 Knapsack problem
# Returns the maximum value that can
# be put in a knapsack of capacity W

from typing import List, Tuple
from examples.knapsack.knapsack import Knapsack
from examples.knapsack.knapsack import KnapsackItem


def dynamic_programming(knapsack: Knapsack) -> Tuple[int, List[KnapsackItem]]:
    W = knapsack.max_weight
    n = knapsack.num_items
    wt = [item.weight for item in knapsack.items]
    val = [item.value for item in knapsack.items]

    dp = [0 for _ in range(W + 1)]

    # To track which items are included
    item_included = [[False] * (W + 1) for _ in range(n)]

    for i in range(1, n + 1):
        # Starting from back, so that we also have data of
        # previous computation when taking i-1 items
        for w in range(W, 0, -1):
            if wt[i - 1] <= w:
                if dp[w - wt[i - 1]] + val[i - 1] > dp[w]:
                    dp[w] = dp[w - wt[i - 1]] + val[i - 1]
                    item_included[i - 1][w] = True

    # To find out which items are included in the optimal solution
    optimal_items = []
    w = W
    for i in range(n - 1, -1, -1):
        if item_included[i][w]:
            optimal_items.append(knapsack.items[i])
            w -= wt[i]

    # Maintains original order
    optimal_items.reverse()

    return dp[W], optimal_items
