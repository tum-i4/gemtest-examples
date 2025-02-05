import typing
from enum import Enum


class HSVColor:

    def __init__(
            self,
            hue_lower: int,
            saturation_lower: int,
            hue_upper: int,
            saturation_upper: int
    ):
        self._hue_lower = hue_lower
        self._saturation_lower = saturation_lower
        self._hue_upper = hue_upper
        self._saturation_upper = saturation_upper

    @property
    def lower_bound(self) -> typing.Tuple[int, int, int]:
        return self._hue_lower, self._saturation_lower, 20

    @property
    def upper_bound(self) -> typing.Tuple[int, int, int]:
        return self._hue_upper, self._saturation_upper, 255


class COLOR_RANGES(typing.Tuple[HSVColor], Enum):
    RED_COLOR = (
        HSVColor(0, 100, 10, 255),
        HSVColor(170, 100, 180, 255)
    )

    WHITE_COLOR = (
        HSVColor(0, 0, 180, 85)
    )

    BLUE_COLOR = (
        HSVColor(50, 100, 130, 225)
    )
