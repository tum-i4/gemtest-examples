from itertools import combinations

from examples.knapsack.knapsack import Knapsack


def brute_force(knapsack: Knapsack):
    max_value = 0
    best_combination = None

    for r in range(len(knapsack.items) + 1):
        for combination in combinations(knapsack.items, r):
            total_weight = sum(item.weight for item in combination)
            total_value = sum(item.value for item in combination)

            if total_weight <= knapsack.max_weight and total_value > max_value:
                max_value = total_value
                best_combination = combination

    return max_value, best_combination
