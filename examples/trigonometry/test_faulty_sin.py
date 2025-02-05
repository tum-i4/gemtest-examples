import math

import pytest

import gemtest as gmt
from gemtest.relations import approximately


class MathLibrary:

    def __init__(self):
        self.__PI = 3.14159265358979323846
        self.__PRECISION = 15

    @staticmethod
    def sin(x: float) -> float:
        x %= 2 * MathLibrary.__PI

        if x < 0:
            return -MathLibrary.sin(-x)

        if x > MathLibrary.__PI:
            return -MathLibrary.sin(x - MathLibrary.__PI)

        assert x >= 0
        assert x <= MathLibrary.__PI

        for i in range(1, MathLibrary.__PRECISION + 1):
            if i % 2 == 0:
                x += math.pow(x, 2 * i + 1) / MathLibrary.factorial(2 * i + 1)
            else:
                x -= math.pow(x, 2 * i + 1) / MathLibrary.factorial(2 * i + 1)

        return x

    @staticmethod
    def factorial(n: int) -> int:
        fact = 1
        for i in range(1, n + 1):
            fact = fact * i
        return fact


data = range(-10, 10)

test_two_pi = gmt.create_metamorphic_relation('Plus 2 π', data, relation=approximately)
test_negate_x = gmt.create_metamorphic_relation('sin(-x)', data)
test_plus_pi = gmt.create_metamorphic_relation('Plus 1 π', data)
test_pi_minus_x = gmt.create_metamorphic_relation('sin(π-x)', data, relation=approximately)


@gmt.transformation(test_two_pi)
@gmt.randomized('n', gmt.RandInt(0, 10))
@gmt.fixed('c', 0)
def two_pi_shift(source_input: float, n: int, c: int) -> float:
    return source_input + 2 * n * math.pi + c


@gmt.transformation(test_negate_x)
def negate(source_input: float) -> float:
    return -source_input


@gmt.transformation(test_plus_pi)
def pi_shift(source_input: float) -> float:
    return source_input + math.pi


@gmt.transformation(test_pi_minus_x)
def pi_shift_minus(source_input: float) -> float:
    return math.pi - source_input


@gmt.relation(test_negate_x, test_plus_pi)
def approximately_negate(source_output: float, followup_output: float) -> bool:
    return approximately(-source_output, followup_output)


@pytest.mark.skip
@gmt.system_under_test()
def test(input_float: float) -> float:
    return MathLibrary.sin(input_float)
