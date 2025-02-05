import math

import gemtest as gmt
from gemtest.relations import approximately

data = range(10)

A = gmt.create_metamorphic_relation(name='A', data=data, relation=approximately)
B = gmt.create_metamorphic_relation(name='B', data=data, relation=approximately)


@gmt.transformation(A)
@gmt.randomized('n', gmt.RandInt(1, 10))
@gmt.fixed('c', 0)
def shift(source_input: float, n: int, c: int) -> float:
    return source_input + 2 * n * math.pi + c


@gmt.general_transformation(B)
@gmt.randomized('n', gmt.RandInt(1, 10))
@gmt.fixed('c', 0)
def shift_complex(mtc: gmt.MetamorphicTestCase, n: int, c: int) -> float:
    # Possibility to manipulate source output f_x for the transformation
    print(f'Access to source output: {mtc.source_output}')
    return mtc.source_input + 2 * n * math.pi + c


@gmt.system_under_test()
def test_cos(input_float: float) -> float:
    return math.cos(input_float)
