import numpy as np

import gemtest as gmt
from gemtest.relations import is_less_than, approximately, or_
from .house_pricing import (
    HousingPricePredictor,
    get_housing_data,
    get_training_and_test_set,
    hide_labels,
)

housing = get_housing_data()
training_set, test_set = get_training_and_test_set(get_housing_data())
p = HousingPricePredictor(training_set)
hide_labels(test_set)  # otherwise this MT would be somewhat pointless

housing_data = [test_set.iloc[n:n + 1] for n in range(20)]

HousePriceTest = gmt.create_metamorphic_relation(
    name='HousePriceTest',
    data=housing_data,
    relation=or_(approximately, is_less_than)
)


@gmt.transformation(HousePriceTest)
@gmt.randomized('increase_rooms_by', gmt.RandInt(1, 10))
def transform(x, increase_rooms_by: int) -> np.ndarray:
    assert all(x["total_rooms"] % 1 == 0)
    copy = x.copy()
    copy["total_rooms"] += increase_rooms_by
    return copy


@gmt.system_under_test(HousePriceTest)
def test_house_pricing_more_rooms(x) -> float:
    assert all(x["total_rooms"] % 1 == 0)
    return p.predict(x).item()
