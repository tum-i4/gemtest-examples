from helper import *


def cube_position(box):
    y_c, x_c, h, w = box
    y1, y2, x1, x2 = int(y_c - h / 2), int(y_c + h / 2), int(x_c - w / 2), int(x_c + w / 2)
    return x1, y1, x2 - x1, y2 - y1


def speckle_augmentation(rad_name, eps=0.1):
    """Add speckle noise to RAD, eps is degree of degradation"""
    aug_name = f"speckle{int(eps * 100)}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    noise = np.random.normal(eps, 1, (256, 256, 64)) + np.random.normal(eps, 1,
                                                                        (256, 256, 64)) * 1j
    return save(np.multiply(rad, noise, dtype=np.complex64), aug_name, rad_name)


def mirror_angle_augmentation(rad_name):
    """Flip RAD along angle dimension"""
    aug_name = "mirror_angle_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    return save(np.flip(rad, [1]), aug_name, rad_name)


def mirror_doppler_augmentation(rad_name):
    """Flip RAD along doppler dimension"""
    aug_name = "mirror_doppler_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    return save(np.flip(rad, [2]), aug_name, rad_name)


def rotate_augmentation(rad_name, shift=120):
    """rotate RAD around the origin, shift is number of buckets"""
    aug_name = f"rotate{shift}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    return save(np.roll(rad, shift, [1]), aug_name, rad_name)


def insert_rad(rad, cube, box_of_dest):
    """helper for delete and insert"""
    box_of_dest[4], box_of_dest[3], box_of_dest[5] = cube.shape
    [x1, x2, y1, y2, z1, z2] = corners_of_box(box_of_dest)
    rad[x1:x2, y1:y2, z1:z2] = cube
    return rad


def insert_augmentation_intersection(rad_name, cube, box_of_dest, dest_name=""):
    """insert cube to location of 'boxOfDest' """
    aug_name = f"insert_intersection{dest_name}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    return save(insert_rad(rad, cube, box_of_dest), aug_name, rad_name)


# rad_name, cube, indexX, indexY, width, height, axis
def insert_augmentation_union(rad_name, rd, ra, box_of_dest, dest_name=""):
    """Inserting a cube along a axis"""
    aug_name = f"insert_union{dest_name}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    [x1, x2, y1, y2, z1, z2] = corners_of_box(box_of_dest)

    rad[x1:x2, y1: y2, :] = ra
    rad[x1:x2, :, z1:z2] = rd

    return save(rad, aug_name, rad_name)


def delete_augmentation_intersection(rad_name, box_of_cube, cube_name=""):
    """delete object at the location 'boxOfCube' """
    aug_name = f"delete{cube_name}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad = read_rad(rad_name)
    cube = box_to_cube(box_of_cube, rad)

    emptyCube = np.zeros_like(cube)
    return save(insert_rad(rad, emptyCube, box_of_cube), aug_name, rad_name)


def delete_augmentation_union(rad_name, box_of_cube, cube_name=""):
    """Deleting of a cube along a axis"""
    aug_name = f"delete_union{cube_name}_"
    if os.path.isfile(f"{PATH_TO_TEST_SET}/RAD/transformed/{aug_name}{rad_name}.npy"):
        return aug_name + rad_name
    rad_RD = np.load(r"./data/test/RAD/cube_library/deleteCube/deleteCubeRD.npy")
    rad_RA = np.load(r"./data/test/RAD/cube_library/deleteCube/deleteCubeRA.npy")
    rad = read_rad(rad_name)
    [x1, x2, y1, y2, z1, z2] = corners_of_box(box_of_cube)

    width, height = int(box_of_cube[3]), int(box_of_cube[4])
    newRACube = np.tile(rad_RA, (width, height, 1))
    rad[x1:x2, y1: y2, :] = newRACube

    width, height = int(box_of_cube[3]), int(box_of_cube[5])
    tmp = np.tile(rad_RD, (height, 1)).transpose()
    newRDCube = np.tile(tmp, (width, 1, 1))
    rad[x1:x2, :, z1:z2] = newRDCube

    return save(rad, aug_name, rad_name)


def left_slice_augmentation(rad_name, cube_name="", eps=0.5, axis=1):
    rad = read_rad(rad_name)

    ones = [[[1 for i in range(64)] for j in range(256)] for k in range(256)]
    noise = np.random.normal(eps, 1, (256, 256, 64)) + \
            np.random.normal(eps, 1, (256, 256, 64)) * 1j

    speckle = np.multiply(ones, noise, dtype=np.complex64)
    speckle_1, _ = np.split(speckle, 2, axis=axis)

    rad_1, _ = np.split(rad, 2, axis=axis)

    rad_1 = np.concatenate((rad_1, speckle_1), axis=axis)

    return save(rad_1, f"slicing_part_1{cube_name}_", rad_name)


def right_slice_augmentation(rad_name, cube_name="", eps=0.5, axis=1):
    rad = read_rad(rad_name)

    ones = [[[1 for i in range(64)] for j in range(256)] for k in range(256)]
    noise = np.random.normal(eps, 1, (256, 256, 64)) + \
            np.random.normal(eps, 1, (256, 256, 64)) * 1j

    speckle = np.multiply(ones, noise, dtype=np.complex64)
    _, speckle_2 = np.split(speckle, 2, axis=axis)

    _, rad_2 = np.split(rad, 2, axis=axis)

    rad_2 = np.concatenate((speckle_2, rad_2), axis=axis)

    return save(rad_2, f"slicing_part_2{cube_name}_", rad_name)
