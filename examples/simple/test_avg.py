import itertools
from typing import TypeVar

import gemtest as gmt
from gemtest.relations import equality

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')


def create_input(a, b, c):
    return {
        "a": a,
        "b": b,
        "c": c
    }


data = [create_input(a, b, c) for a, b, c in list(itertools.product([0, 1], [2, 3], [4, 5]))]

A = gmt.create_metamorphic_relation(name='A', data=data, relation=equality)


@gmt.transformation(A)
def swap(source_input: dict) -> dict:
    return create_input(source_input["c"], source_input["a"], source_input["b"])


@gmt.system_under_test(A)
def test_avg(input_dict: dict) -> float:
    return (input_dict["a"] + input_dict["b"] + input_dict["c"]) / 3
