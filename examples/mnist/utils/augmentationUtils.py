import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf


def get_margins(image):
    rows, cols, channels = image.shape
    img_size = rows  # Assuming a square image
    image = tf.reshape(image, [img_size, img_size])
    nonzero_x_cols = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=0), 0)), tf.int32)
    nonzero_y_rows = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=1), 0)), tf.int32)
    left_margin = tf.reduce_min(nonzero_x_cols)
    right_margin = img_size - tf.reduce_max(nonzero_x_cols) - 1
    top_margin = tf.reduce_min(nonzero_y_rows)
    bot_margin = img_size - tf.reduce_max(nonzero_y_rows) - 1
    return left_margin.numpy(), right_margin.numpy(), top_margin.numpy(), bot_margin.numpy()


def shift(image: np.ndarray, x_range: tuple, y_range: tuple) -> np.ndarray:
    seq = iaa.Sequential([
        iaa.Affine(translate_px={"x": x_range, "y": y_range})])
    aug_image = seq(image=image)
    return aug_image


def scale(image: np.ndarray, width_range: list, height_range: list) -> np.ndarray:
    seq = iaa.Sequential([
        iaa.Affine(scale={"x": (width_range[0], width_range[1]), "y": (height_range[0], height_range[1])})])
    aug_image = seq(image=image)
    return aug_image


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Fliplr()])
    aug_image = seq(image=image)
    return aug_image


def vertical_flip(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Rot90(2)])
    aug_image = seq(image=image)
    return aug_image


def rotate_by_factors_of_90(image: np.ndarray, factor: int) -> np.ndarray:
    seq = iaa.Sequential([iaa.Rot90(factor)])
    aug_image = seq(image=image)
    return aug_image
