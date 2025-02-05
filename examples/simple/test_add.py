import itertools
from typing import TypeVar

import gemtest as gmt
from gemtest.relations import equality

X = TypeVar('X')
Y = TypeVar('Y')


def create_input(x, y):
    return {
        "x": x,
        "y": y
    }


data = [create_input(x, y) for x, y in list(itertools.product([0, 1], [2, 3]))]

B = gmt.create_metamorphic_relation(name='B', data=data, relation=equality)


@gmt.transformation(B)
def swap(source_input: dict) -> dict:
    return create_input(
        x=source_input["y"],
        y=source_input["x"]
    )


@gmt.system_under_test(B)
def test_add(input_dict: dict) -> float:
    return input_dict["x"] + input_dict["y"]
