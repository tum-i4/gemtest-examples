import json
import os

# When running locally
# PATH_TO_RADAR_RETRAINING = "."

# When running on CI/webapp/etc.
PATH_TO_RADAR_RETRAINING = "examples/radar_retraining"

MODEL_NAME = "logs"

PATH_TO_TEST_SET = os.path.join(PATH_TO_RADAR_RETRAINING, "data/test")
PATH_TO_CONFIG = os.path.join(PATH_TO_RADAR_RETRAINING, "model/RADDet/config.json")
PATH_TO_ANCHOR = os.path.join(PATH_TO_RADAR_RETRAINING, "model/RADDet/anchors.txt")
PATH_TO_LOGS = os.path.join(PATH_TO_RADAR_RETRAINING, f"model/RADDet/{MODEL_NAME}/RadarResNet")


def set_config():
    with open(PATH_TO_CONFIG, "r") as jsonFile:
        config = json.load(jsonFile)

    data = config["DATA"]
    data["train_set_dir"] = os.path.join(PATH_TO_RADAR_RETRAINING, "data/train")
    data["test_set_dir"] = PATH_TO_TEST_SET

    config["TRAIN"]["log_dir"] = PATH_TO_LOGS
    config["EVALUATE"]["log_dir"] = PATH_TO_LOGS
    config["INFERENCE"]["log_dir"] = PATH_TO_LOGS

    with open(PATH_TO_CONFIG, "w") as jsonFile:
        json.dump(config, jsonFile, indent=8)


if __name__ == '__main__':
    set_config()
