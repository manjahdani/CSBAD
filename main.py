import yaml
import hydra

from annotation.dataset import build_val_folder
from strategy.select_strategy import build_train_folder

import os
import sys

sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
from ultralytics import YOLO


@hydra.main(version_base=None, config_path="experiments", config_name="experiment")
def main(config):
    # generate validation folder
    build_val_folder(**config.val)

    # generate train folder
    build_train_folder(config.train)

    # init model
    with open(config.model.data, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["path"] = config.dataset.data_path
    with open(config.model.data, mode="w") as f:
        yaml.dump(data, f)
    model = YOLO(config.model.weights, cmd_args=config.model)
    model.train(
        data=config.model.data, epochs=config.model.epochs, batch=config.model.batch
    )


if __name__ == "__main__":
    main()
