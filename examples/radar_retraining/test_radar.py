from glob import glob

import numpy as np
import pytest

import gemtest as gmt
from helper import (
    load_saved_cubes, find_suitable_point, corner_to_whd, get_nearest_box, read_rad)
from model.RADDet.util.helper import iou3d
from radar_augmentation import (
    speckle_augmentation,
    mirror_angle_augmentation,
    mirror_doppler_augmentation,
    rotate_augmentation,
    insert_augmentation_intersection,
    delete_augmentation_intersection,
    left_slice_augmentation,
    right_slice_augmentation
)
from radar_predicter import infer, prepareInference
from set_filepaths import *

"""
This example demonstrates 2 types of MR tests of a Radar data (RAD-Tensor) detector named 'RADDet:
- simple (equality-like) relations that allow for retraining of the model
    - Adding Speckle Noise
    - Mirroring in the angle dimension
    - Mirroring in the doppler dimension
    - Rotating (angle)


- complex relations that don't allow for retraining
    - Deleting of detected objects 
    - Inserting objects 
    - Slicing the Tensor into 2 halves and adding their detections

-  The transformations/augmentations receive the file id as input under which the corresponding 
   RAD Tensor is saved as numpy file.
-  Inference takes file id, loads the file and returns the models prediction
-  The relations take the two predictions and determine pass/fail
"""

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------Preparations------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# Execute tests on small sample set if path_to_rads is None
# Set this path to test a full test set at the given path
path_to_rads = None  # os.path.join(PATH_TO_TESTSET, f"RAD/part_2/*")
set_config()

# Preloading model speeds up execution
model, config = None, None

# Preloading cubes for 'insert_inter' relation
cubes = None

# Dict for the Slice relation
slices = {}


def prepare_test_set():
    global path_to_rads
    if path_to_rads is None:
        test_set = ["000090"]
        # test_set = ["000090", "000156", "000642", "000703", "000877"]
    else:
        test_set = []
        abs_paths = glob(path_to_rads)
        for abs_path in abs_paths:
            test_set.append(abs_path.split("/")[-1].split('.')[0])

    return test_set


@pytest.fixture(scope="session", autouse=True)
def prepare_infer():
    global model
    global config
    if not os.path.exists(os.path.join(PATH_TO_TEST_SET, "RAD/transformed")):
        os.makedirs(os.path.join(PATH_TO_TEST_SET, "RAD/transformed"))
    model, config = prepareInference()


@pytest.fixture(scope="session", autouse=True)
def prepare_cubes():
    global cubes
    cubes = load_saved_cubes()
    return cubes


rad_data = prepare_test_set()

# Initialize all metamorphic relations
speckle = gmt.create_metamorphic_relation(name='speckle', data=rad_data)
mirror_angle = gmt.create_metamorphic_relation(name='mirror_angle', data=rad_data)
mirror_doppler = gmt.create_metamorphic_relation(name='mirror_doppler', data=rad_data)
rotate = gmt.create_metamorphic_relation(name='rotate', data=rad_data)
delete_inter = gmt.create_metamorphic_relation(name='delete_inter', data=rad_data)
insert_inter = gmt.create_metamorphic_relation(name='insert_inter', data=rad_data)
left_slice = gmt.create_metamorphic_relation(name='left_slice', data=rad_data)
right_slice = gmt.create_metamorphic_relation(name='right_slice', data=rad_data)
both_slices = gmt.create_metamorphic_relation(name='both_slices', data=rad_data)


@gmt.transformation(speckle)
@gmt.fixed('dec_eps', 10)
def make_speckle(rad_name, dec_eps):
    return speckle_augmentation(rad_name, eps=dec_eps / 100)


@gmt.transformation(mirror_angle)
def make_mirror_angle(rad_name):
    return mirror_angle_augmentation(rad_name)


@gmt.transformation(mirror_doppler)
def make_mirror_doppler(rad_name):
    return mirror_doppler_augmentation(rad_name)


@gmt.transformation(rotate)
@gmt.fixed('shift', 10)
def make_rotate(rad_name, shift):
    return rotate_augmentation(rad_name, shift=shift)


@gmt.transformation(left_slice)
def make_left_slice(rad_name):
    return left_slice_augmentation(rad_name)


@gmt.transformation(right_slice)
def make_right_slice(rad_name):
    return right_slice_augmentation(rad_name)


@gmt.transformation(both_slices)
def make_both_slices(rad_name):
    global slices
    if rad_name + "L" not in slices:
        slices[rad_name + "L"] = infer(left_slice_augmentation(rad_name), model, config)
    if rad_name + "R" not in slices:
        slices[rad_name + "R"] = infer(right_slice_augmentation(rad_name), model, config)
    return rad_name  # has to be valid name; pred not used later


@gmt.general_transformation(insert_inter)
@gmt.fixed('cube_i', 0)
def make_insert_intersection(mtc: gmt.MetamorphicTestCase, cube_i):
    rad_name = mtc.source_input
    pred = mtc.source_output

    cube_i = cube_i % len(cubes)
    cube = cubes[cube_i]
    dest_box = find_suitable_point(pred, cube.shape)
    if dest_box is None:
        pytest.skip("No suitable insert spot found")
    return insert_augmentation_intersection(rad_name, cube, dest_box, dest_name=f"{cube_i}")


@gmt.general_transformation(delete_inter)
@gmt.fixed('object_i', 0)
def make_delete_intersection(mtc: gmt.MetamorphicTestCase, object_i):
    rad_name = mtc.source_input
    pred = mtc.source_output

    object_i = object_i % len(pred)
    return delete_augmentation_intersection(rad_name, pred[object_i], cube_name=f"{object_i}")


# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------Relations---------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# BASE RELATIONS

@gmt.relation(speckle)
def base_equality_relation(source_pred, followup_pred):
    # return number_objects_less_equal(source_pred, followup_pred)
    # return number_objects_equal(source_pred, followup_pred)
    # return box_centers_approx_equal(source_pred, followup_pred)
    return iou_threshold(source_pred, followup_pred, threshold=0.3)


def number_objects_equal(source_pred, followup_pred):
    return len(source_pred) == len(followup_pred)


def number_objects_less_equal(source_pred, followup_pred):
    return len(source_pred) <= len(followup_pred)


def box_centers_approx_equal(source_pred, followup_pred, tolerance=0.2):
    for s_box in source_pred:
        min_f_box = get_nearest_box(s_box, followup_pred)
        if s_box[:3] != pytest.approx(min_f_box[:3], rel=tolerance):
            return False
    return True


def iou_threshold(source_pred, followup_pred, threshold=0.3):
    # doesn't check if len() are equal
    for s_box in source_pred:
        min_f_box = get_nearest_box(s_box, followup_pred)
        iou = iou3d(s_box[:6], min_f_box[:6], [256, 256, 64])
        if min_f_box is None or iou < threshold:
            print(f"IoU={iou} is below threshold={threshold}")
            print(f"{s_box}\n-----\n{min_f_box}\n-----\n{followup_pred}")
            return False
    return True


def detect_ghost_objects_left_slice(followup_pred, buffer=35):
    for f_box in followup_pred:
        if f_box[1] > 128 + buffer:
            return False
    return True


def detect_ghost_objects_right_slice(followup_pred, buffer=35):
    for f_box in followup_pred:
        if f_box[1] <= 128 - buffer:
            return False
    return True


# METAMORPHIC RELATIONS

@gmt.relation(mirror_angle)
def mirror_angle_relation(source_pred, followup_pred):
    for box in followup_pred:
        box[1] = 256 - box[1]
    return base_equality_relation(source_pred, followup_pred)


@gmt.relation(mirror_doppler)
def mirror_doppler_relation(source_pred, followup_pred):
    for box in followup_pred:
        box[2] = 64 - box[2]
    return base_equality_relation(source_pred, followup_pred)


@gmt.relation(rotate)
@gmt.fixed('shift', 10)
def rotate_relation(source_pred, followup_pred, shift):
    for box in followup_pred:
        box[1] = box[1] + shift
    return base_equality_relation(source_pred, followup_pred)


# @complex_relation(insert_inter)
def insert_equality_relation(o_pred, f_pred, o_rad_name, f_rad_name):
    o_rad = read_rad(o_rad_name[0])  # o_rad_name is passed as tuple for some reason
    f_rad = read_rad(f_rad_name)
    diff = np.subtract(o_rad, f_rad)
    nz = np.nonzero(diff)
    corners = [np.min(nz[0]), np.max(nz[0]), np.min(nz[1]), np.max(nz[1]), np.min(nz[2]),
               np.max(nz[2])]
    # Specifically chosen numbers for cube 2 (as per predictions)
    new_box = np.append(corner_to_whd(corners), [0.83, 5])
    # new_box[3], new_box[4], new_box[5] = 19, 17, 8
    if len(o_pred.shape) < 2:
        o_pred = [o_pred]
    new_o_pred = np.append(o_pred, [new_box], axis=0)
    return base_equality_relation(new_o_pred, f_pred)


@gmt.general_relation(left_slice)
def detected_objects_for_slicing_left(mtc: gmt.MetamorphicTestCase):
    # o_rad_name = mtc.source_input[0]
    o_rad_name = mtc.source_inputs[0]
    f_pred = mtc.followup_outputs[0]

    global slices
    slices[o_rad_name + "L"] = f_pred
    return detect_ghost_objects_left_slice(f_pred)


@gmt.general_relation(right_slice)
def detected_objects_for_slicing_right(mtc: gmt.MetamorphicTestCase):
    o_rad_name = mtc.source_inputs[0]
    f_pred = mtc.followup_outputs[0]

    global slices
    slices[o_rad_name + "R"] = f_pred
    return detect_ghost_objects_right_slice(f_pred)


# source_output, followup_output, source_input, followup_input
# o_pred, f_pred, o_rad_name, f_rad_name

@gmt.general_relation(both_slices)
def both_slices_add_up(mtc: gmt.MetamorphicTestCase):
    o_rad_name = mtc.source_inputs[0]
    o_pred = mtc.source_outputs[0]

    assert o_rad_name + "L" in slices and o_rad_name + "R" in slices
    both_pred = np.concatenate((slices[o_rad_name + "L"], slices[o_rad_name + "R"]), axis=0)
    return iou_threshold(o_pred, both_pred, 0.3)


@gmt.relation(insert_inter)
def insert_subset_relation(o_pred, f_pred):
    # room for improvement
    return (len(f_pred) - len(o_pred)) == 1


def delete_equality_relation(o_pred, f_pred):
    o_pred = o_pred[1:]
    return base_equality_relation(o_pred, f_pred)


@gmt.relation(delete_inter)
def delete_subset_relation(o_pred, f_pred):
    # room for improvement
    return (len(o_pred) - len(f_pred)) == 1


# ---------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------System----------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@gmt.system_under_test()
def test_radar(rad_name):
    """Predict the bounding-boxes and classes in a RAD tensor with the corresponding rad_name
    in the test folder"""
    # returns predictions like: [[x, y, z, w, h, d, score, class_index], ... , ... ]
    return infer(rad_name, model, config)
