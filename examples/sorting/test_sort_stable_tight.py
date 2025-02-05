import random
from collections import Counter

import gemtest as gmt
from data_generator import generate_data_with_identifiers
from stable_sorting_algorithms.insertion_sort import insertionSort_tuple
from stable_sorting_algorithms.merge_sort import mergeSort_tuple
from stable_sorting_algorithms.radix_sort import radixSort

# Generate a list of lists by running the data generation function 20 times
generated_data = [generate_data_with_identifiers() for _ in range(20)]

scaling_tight = gmt.create_metamorphic_relation(
    "scaling_tight", data=generated_data
)  # stable tight
duplicate_element_stable = gmt.create_metamorphic_relation(
    "duplicate_element_stable", data=generated_data
)  # stable tight
sublist_stable = gmt.create_metamorphic_relation(
    "sublist_stable", data=generated_data
)  # stable tight
double_sort = gmt.create_metamorphic_relation(
    "double_sort", data=generated_data
)  # All of them
remove_element_tight = gmt.create_metamorphic_relation(
    "remove_element_tight", data=generated_data
)  # stable tight and unststable tight
bloated_tight = gmt.create_metamorphic_relation(
    "bloated_tight", data=generated_data
)  # stable tight and unststable tight
reverse = gmt.create_metamorphic_relation("reverse", data=generated_data)


@gmt.general_transformation(reverse)
def reverse_list(mtc: gmt.MetamorphicTestCase):
    """
    Reverse the original list
    """
    source_list = mtc.source_input
    reversed_list = source_list[::-1]
    return [reversed_list]


@gmt.general_transformation(scaling_tight)
@gmt.randomized("scalar", gmt.RandInt(1, 10))
def scale_list(mtc: gmt.MetamorphicTestCase, scalar):
    """
    Scales the source list with a scalar.
    """
    source_list = mtc.source_input
    scaled_list = [(value * scalar, identifier) for value, identifier in source_list]
    return [scaled_list]


@gmt.general_transformation(duplicate_element_stable)
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


@gmt.general_transformation(sublist_stable)
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


@gmt.general_relation(reverse)
def reverse_tight_oracle(mtc: gmt.MetamorphicTestCase):
    """
    For the two sorted lists, the reversed input should have all the identifiers in reversed
    order compared to the original list
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    # Check value order equality first
    source_values = [value for value, _ in source_output]
    followup_values = [value for value, _ in followup_output]

    if source_values != followup_values:
        return False

    # Find duplicate values and get the order of identifiers for them
    source_dict = {}
    followup_dict = {}

    for value, identifier in source_output:
        if value not in source_dict:
            source_dict[value] = []
        source_dict[value].append(identifier)

    for value, identifier in followup_output:
        if value not in followup_dict:
            followup_dict[value] = []
        followup_dict[value].append(identifier)

    # Reversed order followup identifiers need to be equal to the source ouput identifiers.
    for value in source_dict:
        if source_dict[value] != followup_dict.get(value, [])[::-1]:
            return False

    return True


@gmt.general_relation(scaling_tight)
def equal_scaled_order_tight_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Checks for equality of identifiers similar to above function while also verifying
    that the source values have been scaled by the exact amount
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output
    scalar = mtc.parameters["scalar"]

    source_values = [item[0] for item in source_output]
    source_identifiers = [item[1] for item in source_output]

    followup_values = [item[0] for item in followup_output]
    followup_identifiers = [item[1] for item in followup_output]

    # Check if the values in followup_output are correctly scaled versions of source_output values
    scaled_source_values = [value * scalar for value in source_values]

    return (
            source_identifiers == followup_identifiers
            and scaled_source_values == followup_values
    )


@gmt.general_relation(duplicate_element_stable)
def equal_order_duplicate_tight_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Identifies the value of the duplicate and checks if the identifiers match exactly
    Holds only for stable algorithms
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    followup_values = [value for value, identifier in followup_output]

    duplicate_value = None

    for value, identifier in followup_output:
        if identifier == "duplicate":
            duplicate_value = value
            break

    source_duplicate_values = [
        (value, identifier)
        for value, identifier in source_output
        if value == duplicate_value
    ]

    followup_values = [
        (value, identifier)
        for value, identifier in followup_output
        if value == duplicate_value
    ]

    source_duplicate_values.append((duplicate_value, "duplicate"))

    return followup_values == source_duplicate_values


@gmt.general_relation(sublist_stable)
def equal_order_sublist_tight_oracle(mtc: gmt.MetamorphicTestCase):
    """
    Checks if values in sublist appear in the same order as the original list.
    Additionally verifies stability
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    source_pairs = [(item[0], item[1]) for item in source_output]
    followup_pairs = [(item[0], item[1]) for item in followup_output]

    def is_sublist_order_preserved(source, sublist):
        # Checks if the order of the elements in sublist and source list is the same
        it = iter(source)
        return all(pair in it for pair in sublist)

    return is_sublist_order_preserved(source_pairs, followup_pairs)


@gmt.general_relation(remove_element_tight)
def tight_oracle_removed_element(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that the element with the specified identifier gets
    removed and the output is exactly the same otherwise.
    """
    source_output = mtc.source_output
    followup_output = mtc.followup_output

    id = mtc.parameters["id_to_remove"]

    identifier_to_remove = f"id_{id}"

    modified_source_output = [
        item for item in source_output if item[1] != identifier_to_remove
    ]

    return modified_source_output == followup_output


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
