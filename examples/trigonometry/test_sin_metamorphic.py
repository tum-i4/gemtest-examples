import math

import gemtest as gmt

mr_1 = gmt.create_metamorphic_relation(name='periodicity', data=range(100))


@gmt.transformation(mr_1)
def plus_two_pi(source_input):
    return source_input + 2 * math.pi


@gmt.relation(mr_1)
def equals(source_output, followup_output):
    return gmt.approximately(source_output, followup_output)


@gmt.system_under_test()
def test_sin(input):
    return math.sin(input)
