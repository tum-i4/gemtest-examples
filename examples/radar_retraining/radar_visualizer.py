from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import model.RADDet.util.drawer as drawer
import model.RADDet.util.helper as helper
import model.RADDet.util.loader as loader
import radar_predicter
from set_filepaths import *


def cutImage(image_name, infer=False):
    image = cv2.imread(image_name)
    if infer:
        part_2 = image[:, 3800:4350, :]
        part_3 = image[:, 5950:6620, :]
        new_img = np.concatenate([part_2, part_3], axis=1)

    else:
        part_2 = image[:, 3950:4250, :]  # 300
        part_3 = image[:, 5750:6500, :]  # 750
        part_4 = image[:, 7450:8850, :]  # 1400
        new_img = np.concatenate([part_2, part_3, part_4], axis=1)

    cv2.imwrite(image_name, new_img)
    return new_img


def processHelper(rad, config_radar, interpolation):
    RA = helper.getLog(helper.getSumDim(helper.getMagnitude(rad, power_order=2),
                                        target_axis=-1), scalar=10, log_10=True)
    RD = helper.getLog(helper.getSumDim(helper.getMagnitude(rad, power_order=2),
                                        target_axis=1), scalar=10, log_10=True)

    # NOTE: change the interval number if high resolution is needed for Cartesian
    RA_cart = helper.toCartesianMask(RA, config_radar,
                                     gapfill_interval_num=interpolation)

    RA_img = helper.norm2Image(RA)[..., :3]
    RD_img = helper.norm2Image(RD)[..., :3]
    RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

    return RA_img, RD_img, RA_cart_img


def process(rad_filename, config_radar, axes, interpolation=15, infer=False, model=None,
            config=None):
    if not os.path.exists("./images/samples/"):
        os.makedirs("./images/samples/")
    RAD = loader.readRAD(rad_filename)

    if RAD is not None:
        RA_img, RD_img, RA_cart_img = processHelper(RAD, config_radar, interpolation)

        drawer.clearAxes(axes)

        if infer:
            assert model is not None and config is not None
            nms_pred = radar_predicter.infer("", model, config, RAD)
            processInference(RD_img, RA_img, RA_cart_img, nms_pred, axes)
            drawer.imgPlot(RD_img, axes[1], None, 1, "RD")
            drawer.imgPlot(RA_img, axes[2], None, 1, "RA")

        else:
            drawer.imgPlot(RD_img, axes[1], None, 1, "RD")
            drawer.imgPlot(RA_img, axes[2], None, 1, "RA")
            drawer.imgPlot(RA_cart_img, axes[3], None, 1, "Cartesian")

        suffix = "_pred.png" if infer else ".png"
        drawer.saveFigure("./images/samples/",
                          rad_filename.split("/")[-1].split(".")[0] + suffix)
        return cutImage(
            "./images/samples/" + rad_filename.split("/")[-1].split(".")[0] + suffix,
            infer=infer)


def processInference(RD_img, RA_img, RA_cart_img, radar_nms_pred, axes):
    """ draw only boxes on the input images """
    all_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    colors = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
              (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    for i in range(len(radar_nms_pred)):
        bbox3d = radar_nms_pred[i, :6]
        cls = int(radar_nms_pred[i, 7])
        color = colors[int(cls)]
        # draw box
        mode = "box"  # either "box" or "ellipse"
        # boxes information added in the ground truth dictionary
        RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
        RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
        # draw boxes
        drawer.drawBoxOrEllipse(RD_box, all_classes[cls], axes[1], color,
                                x_shape=RD_img.shape[1], mode=mode)
        drawer.drawBoxOrEllipse(RA_box, all_classes[cls], axes[2], color,
                                x_shape=RA_img.shape[1], mode=mode)


def visualizeCompare(rad_name="*_*", folder="*"):
    if not os.path.exists("./images/comparison/"):
        os.makedirs("./images/comparison/")
    path = f"RAD/{folder}/{rad_name}.npy"
    config = loader.readConfig(PATH_TO_CONFIG)
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
    model, config = radar_predicter.prepareInference()
    all_augmented_names = glob(os.path.join(config_data["test_set_dir"], path))
    originals = {}
    for aug_name in tqdm(all_augmented_names):
        aug_img = process(
            rad_filename=aug_name,
            config_radar=config_radar,
            axes=axes,
            interpolation=15,
            infer=True,
            model=model,
            config=config
        )
        orig_name = aug_name[-10:]
        if orig_name in originals:
            orig_img = originals[orig_name]
        else:
            orig_path = glob(os.path.join(config_data["test_set_dir"], f"RAD/*/{orig_name}"))[
                0]
            orig_img = process(
                rad_filename=orig_path,
                config_radar=config_radar,
                axes=axes,
                interpolation=15,
                infer=True,
                model=model,
                config=config
            )
            originals[orig_name] = orig_img
        new_img = np.concatenate([aug_img, orig_img], axis=0)
        cv2.imwrite(f"./images/comparison/{aug_name.split('/')[-1].split('.')[0]}_comp.png",
                    new_img)


def main(path="RAD/transformed/*.npy", infer=False):
    config = loader.readConfig(PATH_TO_CONFIG)
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    if infer:
        fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
    else:
        fig, axes = drawer.prepareFigure(4, figsize=(100, 8))

    interpolation = 15

    if not os.path.exists("./images/samples/"):
        os.makedirs("./images/samples/")

    all_RAD_files = glob(os.path.join(config_data["test_set_dir"], path))

    if infer:
        model, config = radar_predicter.prepareInference()
    else:
        model, config = None, None

    for i in tqdm(range(len(all_RAD_files))):
        RAD_filename = all_RAD_files[i]
        process(
            rad_filename=RAD_filename,
            config_radar=config_radar,
            axes=axes,
            interpolation=interpolation,
            infer=infer,
            model=model,
            config=config
        )


if __name__ == "__main__":
    # main("RAD/*/000703.npy", infer=True)
    visualizeCompare(rad_name="*", folder="*")
