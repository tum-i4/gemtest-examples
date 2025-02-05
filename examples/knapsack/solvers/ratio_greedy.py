from examples.knapsack.knapsack import Knapsack


def greedy(knapsack: Knapsack):
    sorted_items = sorted(knapsack.items, key=lambda element: element.value / element.weight,
                          reverse=True)
    current_weight = 0
    optimal_value = 0
    optimal_items = []

    for item in sorted_items:
        if current_weight + item.weight <= knapsack.max_weight:
            optimal_items.append(item)
            current_weight += item.weight
            optimal_value += item.value

    return optimal_value, optimal_items
