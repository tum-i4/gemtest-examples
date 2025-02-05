import os
import uuid

import numpy as np
import pytest
from PIL import Image

import gemtest as gmt
from examples.gtsrb.utils.deprecated_speed_sign.operation import write_speed_operation
from examples.gtsrb.utils.domain.dataset import gtsrb_load_test_default, \
    gtsrb_dataset, gtsrb_load_train_default
from examples.gtsrb.utils.domain.image import rgb_image
from examples.gtsrb.utils.domain.model import lenet5_model

# Set to an integer to fail the test if the pipeline generates less
# Value of None will
_SUCCESS_THRESHOLD = None


@pytest.mark.skip(reason="Test broken.")
@pytest.mark.parametrize("speed_text", [
    "20", "110", "30", "130", "90", "50"
])
def test_write_speed_operation(speed_text: str):
    try:
        train_strategy = gtsrb_load_train_default.GTSRBLoadTrainDefault(
            seeked_classes=[15],
            gtsrb_training_glob="../../assets/GTSRB_Images/Train/*/*",
        )
        test_strategy = gtsrb_load_test_default.GTSRBLoadTestDefault(
            seeked_classes=[15],
            gtsrb_testing_glob="../../assets/GTSRB_Images/Test/*",
            gtsrb_testing_ground_truth_path="../../assets/GTSRB_Images/Test.csv",
        )
        gtsrb_ds = gtsrb_dataset.GTSRBDataset(
            train_load_strategy=train_strategy,
            test_load_strategy=test_strategy
        )
        model = lenet5_model.Lenet5PytorchModelAdapter(
            '../../assets/lenet5_self_trained.pth'
        )
        valid_count = 0
        for img, _ in gtsrb_ds.train_iterator():
            img_np = np.array(img)
            rgb_img = rgb_image.RGBImage(img_np)
            result = write_speed_operation.WriteSpeedOperation(
                speed_text=speed_text,
                # Font sourced from http://www.peter-wiegel.de/DinBreit.html
                font_path="../assets/din1451alt.ttf",
                control_image_path="../../../assets/forbidden.png"
            ).apply(rgb_img)
            if result is not None:
                valid_count += 1
                result.plot()
                os.makedirs(f"../../outputs/speeds/{speed_text}/images", exist_ok=True)
                os.makedirs(f"../../outputs/speeds/{speed_text}/latents", exist_ok=True)
                fn = uuid.uuid1()
                Image.fromarray(result.value, 'RGB').save(
                    f"../../outputs/speeds/{speed_text}/images/{fn}.png"
                )
                latent_t = model.get_latent_representation(
                    Image.fromarray(result.value, 'RGB'))
                np.save(f"../../outputs/speeds/{speed_text}/latents/{fn}.npy", latent_t)
            if _SUCCESS_THRESHOLD is not None and valid_count == _SUCCESS_THRESHOLD:
                break
        if _SUCCESS_THRESHOLD is None:
            assert valid_count > 0
        else:
            assert valid_count == _SUCCESS_THRESHOLD
    except ValueError:
        gmt.skip("GTSRB_Images dataset not found in 'assets' folder")
