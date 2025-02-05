import random
from collections import Counter

import gemtest as gmt
from data_generator import generate_data_with_identifiers
from unstable_sorting_algorithms.heap_sort import heapSort
from unstable_sorting_algorithms.quicksort import quickSort
from unstable_sorting_algorithms.selection_sort import selectionSort

# Generate a list of lists by running the data generation function 20 times
generated_data = [generate_data_with_identifiers() for _ in range(20)]

# Creation of MRs, set the parameter dict for the scale relation
parameters_dict = {
    "scalar": [random.randint(1, 10) for _ in range(2)],
    "bloat": [random.randint(1, 5) for _ in range(2)],
    "remove_index": [f"id_{random.randint(0, 19)}"],
}

permutation = gmt.create_metamorphic_relation(
    "permutation", data=generated_data
)  # unstable and unstable tight
sublist = gmt.create_metamorphic_relation("sublist", data=generated_data)
double_sort = gmt.create_metamorphic_relation(
    "double_sort", data=generated_data
)  # All of them
remove_element_tight = gmt.create_metamorphic_relation(
    "remove_element_tight", data=generated_data, parameters=parameters_dict
)  # stable tight and unststable tight
bloated_tight = gmt.create_metamorphic_relation(
    "bloated_tight", data=generated_data, parameters=parameters_dict
)  # stable tight and unststable tight


@gmt.general_transformation(permutation)
def shuffle_list(mtc: gmt.MetamorphicTestCase):
    """
    Create random permutation of the original list
    """
    source_list = mtc.source_input
    shuffled_list = source_list[:]
    random.shuffle(shuffled_list)
    return [shuffled_list]


@gmt.general_transformation(sublist)
def extract_random_sublist(mtc: gmt.MetamorphicTestCase):
    """
    Extracts a random sublist from the original source list.
    """
    source_list = mtc.source_input

    sublist_size = random.randint(1, len(source_list))

    start_index = random.randint(0, len(source_list) - sublist_size)
    sublist = source_list[start_index: start_index + sublist_size]

    return [sublist]


@gmt.general_transformation(double_sort)
def sort_source_input(mtc: gmt.MetamorphicTestCase):
    """
    Sorts the source input
    """
    sorted_input = sorted(mtc.source_input, key=lambda x: x[0])

    return [sorted_input]


@gmt.general_transformation(remove_element_tight)
@gmt.randomized("id_to_remove", gmt.RandInt(0, 19))
def remove_random_element(mtc: gmt.MetamorphicTestCase, id_to_remove: int):
    """
    Removes a random element from the source input list
    """
    identifier_to_remove = f"id_{id_to_remove}"

    source_input = mtc.source_input

    modified_input = [item for item in source_input if item[1] != identifier_to_remove]

    return [modified_input]


@gmt.general_transformation(bloated_tight)
def bloat_list(mtc: gmt.MetamorphicTestCase):
    """
    Bloats up the source input list a random number of times.
    """
    n = mtc.parameters["bloat"]

    source_input = mtc.source_input

    bloated_list = source_input * n

    return [bloated_list]


# Metamorphic Relations
@gmt.general_relation(permutation, double_sort)
def equal_order_weak_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Extracts values of source and followup output and checks for equality
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_values = [item[0] for item in source_output]
    followup_values = [item[0] for item in followup_output]

    return source_values == followup_values


@gmt.general_relation(sublist)
def equal_order_sublist_weak_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Checks if values in sublist appear in the same order as the original list
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_values = [item[0] for item in source_output]
    followup_values = [item[0] for item in followup_output]

    def is_sublist_order_preserved(source, sublist):
        # Checks if the order of the elements in sublist and source list is the same
        it = iter(source)
        return all(item in it for item in sublist)

    return is_sublist_order_preserved(source_values, followup_values)


@gmt.general_relation(remove_element_tight)
def tight_oracle_removed_element(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that the element with the specified identifier gets
    removed and the output values are the same otherwise.
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    id = mtc.parameters["id_to_remove"]

    identifier_to_remove = f"id_{id}"

    modified_source_values = [
        item[0] for item in source_output if item[1] != identifier_to_remove
    ]

    followup_values = [item[0] for item in followup_output]

    # Check if the values match
    return modified_source_values == followup_values


@gmt.general_relation(bloated_tight)
def equal_duplicates(mtc: gmt.MetamorphicTestCase):
    """
    Verifies order and additionally checks that the quantity of each value
    is exactly scaled by the bloat_factor specified.
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    bloat_factor = mtc.parameters["bloat"]

    source_counter = Counter(source_output)
    followup_counter = Counter(followup_output)

    # Verify that each unique item in source_output appears exactly bloat_factor times in followup_output
    for item in source_counter:
        if followup_counter[item] != source_counter[item] * bloat_factor:
            return False
    return True


# Unstable
@gmt.system_under_test()
def test_heapSort(list):
    return heapSort(list)


@gmt.system_under_test()
def test_quickSort(list):
    return quickSort(list, 0, len(list) - 1)


@gmt.system_under_test()
def test_selectionSort(list):
    return selectionSort(list, len(list))
