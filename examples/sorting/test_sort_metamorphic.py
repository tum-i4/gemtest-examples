import gemtest as gmt
from data_generator import generate_data
from stable_sorting_algorithms.insertion_sort import insertionSort
from stable_sorting_algorithms.merge_sort import mergeSort

# Generate a list of lists by running the data generation function 20 times
generated_data = [generate_data(n=5, min_value=-100, max_value=100) for _ in range(20)]

mr1 = gmt.create_metamorphic_relation(
    "add_element", data=generated_data, number_of_sources=2)


@gmt.general_transformation(mr1)
@gmt.randomized("s", gmt.RandInt(1, 10))
def add_element(mtc: gmt.MetamorphicTestCase, s: int):
    a, b = mtc.source_inputs
    e = max(a + b) + s
    return a + b, [e] + a + b


@gmt.general_relation(mr1)
def correct_sorted(mtc: gmt.MetamorphicTestCase):
    a, b = mtc.source_inputs
    c, d = mtc.followup_outputs
    e = max(a + b) + mtc.parameters['s']
    return all(x == y for x, y in zip(c + [e], d))


@gmt.system_under_test()
def test_insertionSort(list):
    return insertionSort(list)


@gmt.system_under_test()
def test_mergeSort(list):
    return mergeSort(list)
