import pytest

from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.triangle_sign.triangle_transformation import TriangleMorph
from gemtest import load_image_resource


def test_triangle_sign_operation():
    wrong_sign = load_image_resource("dummy_data/00000.ppm")
    triangle1 = load_image_resource("dummy_data/00006.ppm")
    triangle2 = load_image_resource("dummy_data/00008.ppm")
    wrong_img = RGBImage(wrong_sign)
    triangle1 = RGBImage(triangle1)
    triangle2 = RGBImage(triangle2)

    assert TriangleMorph(triangle1).apply(triangle2)
    assert TriangleMorph(triangle2).apply(triangle1)
    with pytest.raises(Exception):
        TriangleMorph(wrong_img).apply(triangle1)
    with pytest.raises(Exception):
        TriangleMorph(wrong_img).apply(triangle2)
    with pytest.raises(Exception):
        TriangleMorph(wrong_img).apply(wrong_img)
    with pytest.raises(Exception):
        TriangleMorph(triangle1).apply(wrong_img)
    with pytest.raises(Exception):
        TriangleMorph(triangle2).apply(wrong_img)
