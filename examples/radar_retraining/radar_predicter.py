from glob import glob

import tensorflow as tf
import tensorflow.keras as K

import model.RADDet.model.model as M
import model.RADDet.util.helper as helper
import model.RADDet.util.loader as loader
from set_filepaths import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prepareInference():
    # COPIED FROM RADDet.inference !!!

    # NOTE: GPU manipulation, you may print this out if necessary
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    config = loader.readConfig(config_file_name=PATH_TO_CONFIG)
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_inference = config["INFERENCE"]

    anchor_boxes = loader.readAnchorBoxes(
        anchor_boxes_file=PATH_TO_ANCHOR)  # load anchor boxes with order

    # NOTE: using the yolo head shape out from model for data generator
    model = M.RADDet(config_model, config_data, config_train, anchor_boxes)
    model.build([None] + config_model["input_shape"])
    model.backbone_stage.summary()
    model.summary()

    # NOTE: RAD Boxes ckpt
    logdir = os.path.join(config_inference["log_dir"],
                          "b_" + str(config_train["batch_size"]) + \
                          "lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    optimizer = K.optimizers.Adam(learning_rate=config_train["learningrate_init"])
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_steps)
    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(logdir, "ckpt"), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored RAD Boxes Model from {}".format(manager.latest_checkpoint))
    else:
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")

    return model, config


def prepareData(rad_complex, config_data):
    RAD_data = helper.complexTo2Channels(rad_complex)
    RAD_data = (RAD_data - config_data["global_mean_log"]) / \
               config_data["global_variance_log"]
    data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
    return data


def infer(to_infer_name, model=None, config=None, rad=None):
    if model is None or config is None:
        model, config = prepareInference()

    config_data = config["DATA"]
    config_evaluate = config["EVALUATE"]
    config_inference = config["INFERENCE"]
    config_model = config["MODEL"]

    if rad is None:
        RAD_file = \
            glob(os.path.join(config_data["test_set_dir"], f"RAD/*/{to_infer_name}.npy"))[0]
        RAD_complex = loader.readRAD(RAD_file)
    else:
        RAD_complex = rad

    data = prepareData(RAD_complex, config_data)

    if data is None:
        return []

    feature = model(data)
    pred_raw, pred = model.decodeYolo(feature)
    pred_frame = pred[0]
    predictions = helper.yoloheadToPredictions(pred_frame, conf_threshold=config_evaluate[
        "confidence_threshold"])
    nms_pred = helper.nms(predictions,
                          config_inference["nms_iou3d_threshold"],
                          config_model["input_shape"],
                          sigma=0.3, method="nms")

    return nms_pred


if __name__ == "__main__":
    model, config = prepareInference()
    print(infer("000090", model, config))
