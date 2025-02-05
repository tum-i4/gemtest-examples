import pickle
from glob import glob

import numpy as np

from model.RADDet.util.helper import iou3d
from set_filepaths import *


def save(rad, aug_name, file_name):
    path = f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{file_name}.npy"
    np.save(path, rad)
    return aug_name + file_name


def read_rad(file_name):
    path = glob(f"{PATH_TO_TEST_SET}/RAD/*/{file_name}.npy")[0]
    if os.path.exists(path):
        return np.load(path)
    else:
        return None


def corners_of_box(box):
    # [x, y, z, w, h, d, score, class] #x1, x2, y1, y2, z1, z2
    if len(box) == 8:
        [x_c, y_c, z_c, h, w, d, _, _] = box
    elif len(box) == 6:
        [x_c, y_c, z_c, h, w, d] = box
    else:
        raise ValueError("Coordinate box of invalid shape")

    return [int(x_c - w / 2), int(x_c + w / 2),
            int(y_c - h / 2), int(y_c + h / 2),
            int(z_c - d / 2), int(z_c + d / 2)]


def box_to_cube(box, rad):
    """range(from top):x ; angle:y ; doppler: z"""
    b = corners_of_box(box)
    cube = rad[
           b[0]: b[1],
           b[2]: b[3],
           b[4]: b[5]]
    return cube


def load_saved_cubes():
    cubes = []
    files = glob(f"{PATH_TO_TEST_SET}/RAD/cube_library/insertCube/*.npy")
    for f in files:
        cube = np.load(f)
        cubes.append(cube)
    return cubes


def create_insertion_cubes():
    # Could be BROKEN! Fix/Test before usage
    # gts and RAD have to correspond glob may not be in right order
    gt_files = glob(f"{PATH_TO_TEST_SET}/gt/part1/*")
    gts = []
    for gt_file in gt_files:
        with open(gt_file, "rb") as f:
            gt = pickle.load(f)
            gts.append(gt["boxes"])

    rad_files = glob(f"{PATH_TO_TEST_SET}/RAD/part1/*")
    rads = []
    for rad_file in rad_files:
        rad = np.load(rad_file)
        rads.append(rad)

    path_prefix = f"{PATH_TO_TEST_SET}/RAD/cube_library/insertCube"
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    path_prefix = path_prefix + "/cube_"

    for i, rad in enumerate(rads):
        for box in gts[i]:
            cube = box_to_cube(box, rad)
            np.save(path_prefix + f"{i}.npy", cube)
            i += 1


def find_suitable_point(pred, input_shape, tries=100):
    shape_array = [0, 0, 0]
    shape_array[1], shape_array[0], shape_array[2] = input_shape
    buffer_amount = 5
    xy_gap = (max(shape_array[0], shape_array[1]) / 2) + buffer_amount
    z_gap = (shape_array[2] / 2) + buffer_amount
    for _ in range(tries):
        dest_box = np.append(
            np.append(np.random.randint(xy_gap, 256 - xy_gap, 2),
                      np.random.randint(z_gap, 64 - z_gap)), shape_array)
        buffer = [0, 0, 0, buffer_amount, buffer_amount, buffer_amount]
        if 28 <= dest_box[2] <= 36:
            continue
        for pred_box in pred:
            if iou3d(np.add(dest_box, buffer), pred_box[:6], [256, 256, 64]) > 0:
                break
        return dest_box
    return None


def corner_to_whd(corners):
    assert len(corners) == 6
    box = np.zeros(6)
    box[0] = np.round((corners[0] + corners[1]) / 2)
    box[1] = np.round((corners[2] + corners[3]) / 2)
    box[2] = np.round((corners[4] + corners[5]) / 2)
    box[3] = np.round(corners[1] - corners[0])
    box[4] = np.round(corners[3] - corners[2])
    box[5] = np.round(corners[5] - corners[4])
    return box


def get_nearest_box(box1, boxes):
    min_box = None
    min_dist = np.Infinity
    for box2 in boxes:
        dist = np.sum((box1[:3] - box2[:3]) ** 2, axis=0)
        if dist < min_dist:
            min_dist = dist
            min_box = box2
    return min_box
