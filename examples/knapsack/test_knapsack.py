import gemtest as gmt
from knapsack import Knapsack, generate_knapsacks
from solvers import brute_force
from solvers import dynamic_programming

knapsacks = generate_knapsacks(number_of_knapsacks=10, min_weight=1, max_weight=20)

add = gmt.create_metamorphic_relation(name="add", data=knapsacks)
combine = gmt.create_metamorphic_relation(
    name="combine",
    data=knapsacks,
    number_of_sources=2,
    testing_strategy=gmt.TestingStrategy.SAMPLE,
    number_of_test_cases=10
)


@gmt.transformation(add)
@gmt.randomized('items_to_add', gmt.RandInt(1, 10))
def add_items(knapsack: Knapsack, items_to_add: int):
    knapsack.add_items(items_to_add)
    return knapsack


@gmt.relation(add)
def check_add_items(source_output: int, followup_output: int):
    return source_output <= followup_output


@gmt.general_transformation(combine)
def combine_knapsacks(mtc: gmt.MetamorphicTestCase):
    knapsack1, knapsack2 = mtc.source_inputs
    combined_max_weight = knapsack1.max_weight + knapsack2.max_weight
    combined_items = knapsack1.items + knapsack2.items
    return Knapsack(max_weight=combined_max_weight, items=combined_items)


@gmt.general_relation(combine)
def check_combine_knapsacks(mtc: gmt.MetamorphicTestCase):
    return mtc.source_outputs[0] + mtc.source_outputs[1] <= mtc.followup_output


# @gmt.system_under_test(add, combine)
# def test_knapsack_greedy(knapsack: Knapsack):
#      max_value, _ = greedy(knapsack)
#      print(f"max value {max_value}: knapsack {knapsack}")
#      return max_value

@gmt.system_under_test(add, combine)
def test_knapsack_dp(knapsack: Knapsack):
    max_value, _ = dynamic_programming(knapsack)
    return max_value

# Brute Force implementation has exponential time complexity,
# commented out to reduce ci/cd pipeline runtimes

@gmt.system_under_test(add, combine)
def test_knapsack_brute_force(knapsack: Knapsack):
    max_value, selected_items = brute_force(knapsack)
    print(f"max value {max_value}: knapsack {knapsack}")
    return max_value
