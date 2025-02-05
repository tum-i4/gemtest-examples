import math

import gemtest as gmt
from gemtest.relations import approximately

# For each parameter combination in the dict one instance of a metamorphic test case is created
a_params = {
    'n': [3, 7],
    'c': [0, 2 * math.pi]
}

data = range(-10, 10)
# Register a new metamorphic test by specifying its name, datasource and optionally
# a transform, relation and parameters.
A = gmt.create_metamorphic_relation('A', data, relation=approximately)
A_parameters = gmt.create_metamorphic_relation('A_parameters', data, parameters=a_params)
B = gmt.create_metamorphic_relation('B', data)


@gmt.transformation(A)
@gmt.randomized('n', gmt.RandInt(1, 10))
@gmt.fixed('c', 0)
def shift(source_input: float, n: int, c: int) -> float:
    return source_input + 2 * n * math.pi + c


@gmt.general_transformation(A_parameters)
def shift_params(mtc: gmt.MetamorphicTestCase) -> float:
    result = mtc.source_input + 2 * mtc.parameters['n'] * math.pi + mtc.parameters['c']
    return result


@gmt.transformation(B)
def negate(source_input: float) -> float:
    return -source_input


@gmt.general_relation(A_parameters)
def approximately_with_parameters(mtc: gmt.MetamorphicTestCase) -> bool:
    return approximately(mtc.source_output, mtc.followup_output)


@gmt.relation(B)
def approximately_negate(source_output: float, followup_output: float) -> bool:
    return approximately(-source_output, followup_output)


@gmt.system_under_test()
def test_sin(input_float: float, **_kwargs) -> float:
    return math.sin(input_float)
