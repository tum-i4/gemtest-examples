import random
from collections import Counter

import gemtest as gmt
from data_generator import generate_data_with_identifiers
from stable_sorting_algorithms.insertion_sort import insertionSort_tuple
from stable_sorting_algorithms.merge_sort import mergeSort_tuple
from stable_sorting_algorithms.radix_sort import radixSort

# Generate a list of lists by running the data generation function 20 times
generated_data = [generate_data_with_identifiers() for _ in range(20)]

permutation_stable = gmt.create_metamorphic_relation(
    "permutation_stable", data=generated_data
)  # Stable and stable tight
scaling = gmt.create_metamorphic_relation(
    "scaling", data=generated_data
)  # stable and unstable
duplicate_element = gmt.create_metamorphic_relation(
    "duplicate_element", data=generated_data
)  # stable and unstable
sublist = gmt.create_metamorphic_relation("sublist", data=generated_data)
double_sort = gmt.create_metamorphic_relation(
    "double_sort", data=generated_data
)  # All of them
remove_element = gmt.create_metamorphic_relation(
    "remove_element", data=generated_data
)  # stable and unstable
bloated = gmt.create_metamorphic_relation(
    "bloated", data=generated_data
)  # stable and unstable


# Transformation functions to transform source inputs
@gmt.general_transformation(permutation_stable)
def shuffle_list(mtc: gmt.MetamorphicTestCase):
    """
    Create random permutation of the original list
    """
    source_list = mtc.source_input
    shuffled_list = source_list[:]
    random.shuffle(shuffled_list)
    return [shuffled_list]


@gmt.general_transformation(scaling)
@gmt.randomized("scalar", gmt.RandInt(1, 10))
def scale_list(mtc: gmt.MetamorphicTestCase, scalar):
    """
    Scales the source list with a scalar.
    """
    source_list = mtc.source_input
    scaled_list = [(value * scalar, identifier) for value, identifier in source_list]
    return [scaled_list]


@gmt.general_transformation(duplicate_element)
def duplicate_random_element(mtc: gmt.MetamorphicTestCase):
    """
    Duplicates a random item from the list, changing the identifier to 'duplicate' to detect it later on.
    """
    source_list = mtc.source_input

    index_to_duplicate = random.randint(0, len(source_list) - 1)
    item_to_duplicate = source_list[index_to_duplicate]

    # duplicate identifier to later on know which element is the duplicate
    duplicate_item = (item_to_duplicate[0], "duplicate")

    followup_list = source_list[:]
    followup_list.append(duplicate_item)

    return [followup_list]


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


@gmt.general_transformation(remove_element)
@gmt.randomized("id_to_remove", gmt.RandInt(0, 19))
def remove_random_element(mtc: gmt.MetamorphicTestCase, id_to_remove: int):
    """
    Removes a random element from the source input list
    """
    identifier_to_remove = f"id_{id_to_remove}"

    source_input = mtc.source_input

    modified_input = [item for item in source_input if item[1] != identifier_to_remove]

    return [modified_input]


@gmt.general_transformation(bloated)
@gmt.randomized("bloat", gmt.RandInt(1, 5))
def bloat_list(mtc: gmt.MetamorphicTestCase, bloat):
    """
    Bloats up the source input list a random number of times.
    """

    source_input = mtc.source_input

    bloated_list = source_input * bloat

    return [bloated_list]


# Metamorphic Relations
@gmt.general_relation(double_sort)
def equal_order_weak_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Extracts values of source and followup output and checks for equality
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_values = [item[0] for item in source_output]
    followup_values = [item[0] for item in followup_output]

    return source_values == followup_values


@gmt.general_relation(permutation_stable)
def equal_order_tight_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that for each unique value, the order of identifiers is the same
    between input and output within the source and followup pairs separately.
    Ultimately this verifies the stability of the algorithm
    """
    source_input = mtc.source_input
    followup_input = mtc.followup_input

    source_output = mtc.source_output
    followup_output = mtc.followup_output

    # Helper function to group identifiers by value
    def group_by_value(data):
        grouped = {}
        for value, identifier in data:
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(identifier)
        return grouped

    source_input_grouped = group_by_value(source_input)
    source_output_grouped = group_by_value(source_output)
    followup_input_grouped = group_by_value(followup_input)
    followup_output_grouped = group_by_value(followup_output)

    # Check that the order of identifiers is the same for each value in the input-output pairs
    for value, source_identifiers in source_input_grouped.items():
        source_output_identifiers = source_output_grouped.get(value, [])

        if source_identifiers != source_output_identifiers:
            return False

    for value, followup_identifiers in followup_input_grouped.items():
        followup_output_identifiers = followup_output_grouped.get(value, [])

        if followup_identifiers != followup_output_identifiers:
            return False

    return True


@gmt.general_relation(scaling)
def equal_scaled_order_weak_oracle(mtc: gmt.MetamorphicTestCase):
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_identifiers = [item[1] for item in source_output]
    followup_identifiers = [item[1] for item in followup_output]

    return source_identifiers == followup_identifiers


@gmt.general_relation(duplicate_element)
def equal_order_duplicate_weak_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Checks equality of values while ignoring the duplicate entry
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    # Extract values from (value, identifier) tuples, ignoring "duplicate"
    source_values = [value for value, identifier in source_output]
    followup_values = [
        value for value, identifier in followup_output if identifier != "duplicate"
    ]

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


@gmt.general_relation(remove_element)
def exact_removal_relation(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that we have exactly the same amount of values or one less, that is the case
    when the element was removed.
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_counter = Counter(source_output)
    followup_counter = Counter(followup_output)

    return all(
        source_count - followup_counter.get(item, 0) in [0, 1]
        for item, source_count in source_counter.items()
    )


@gmt.general_relation(bloated)
def equal_bloated_order(mtc: gmt.MetamorphicTestCase):
    """
    Gets order of unique elements and compares between source and followup output
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    # Helper function to get a list of unique value orders
    def extract_value_order(output):
        seen_values = set()
        order = []
        for item in output:
            value, _ = item
            if value not in seen_values:
                seen_values.add(value)
                order.append(value)
        return order

    # Extract unique value orders
    source_value_order = extract_value_order(source_output)
    followup_value_order = extract_value_order(followup_output)

    return source_value_order == followup_value_order


# SUT Definitions
# Stable
@gmt.system_under_test()
def test_insertionSort(list):
    return insertionSort_tuple(list)


@gmt.system_under_test()
def test_mergeSort(list):
    return mergeSort_tuple(list)


@gmt.system_under_test()
def test_radixSort(list):
    return radixSort(list)
