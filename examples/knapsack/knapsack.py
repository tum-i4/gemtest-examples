import random
from dataclasses import dataclass
from typing import List


@dataclass
class KnapsackItem:
    weight: int
    value: int


@dataclass
class Knapsack:
    min: int = 0
    max: int = 0
    max_weight: int = None
    items: list[KnapsackItem] = None
    num_items: int = None

    def __post_init__(self):
        if self.max_weight is None and self.items is None:
            # This constructor creates a knapsack with random weight and items
            self.max_weight = random.randint(self.min, self.max)
            self.num_items = random.randint(self.min, self.max)
            self.items = self._generate_items()
        elif self.max_weight is not None and self.items is not None:
            # This constructor creates a knapsack with given max_weight and items
            self.num_items = len(self.items)

    def _generate_items(self):
        return [KnapsackItem(
            random.randint(self.min, self.max),
            random.randint(self.min, self.max)
        ) for _ in range(self.num_items)]

    def add_items(self, num_items: int):
        for _ in range(num_items):
            self.items.append(KnapsackItem(
                random.randint(self.min, self.max),
                random.randint(self.min, self.max)
            ))

    def __str__(self):
        item_details = ", ".join([f"({item.weight}, {item.value})" for item in self.items])
        return f"Max Weight: {self.max_weight}: {self.num_items} items: [{item_details}]"


def generate_knapsacks(number_of_knapsacks: int, min_weight: int,
                       max_weight: int) -> List[Knapsack]:
    knapsacks = []
    for _ in range(number_of_knapsacks):
        knapsacks.append(Knapsack(min=min_weight, max=max_weight))
    return knapsacks
