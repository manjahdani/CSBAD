import os
import sys
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np

from annotation.dataset import build_val_folder
from strategy.select_strategy import build_train_folder

sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
from ultralytics import YOLO


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(version_base=None, config_path="experiments", config_name="experiment")
def main(config):
    set_random(config.seed)

    # generate validation folder
    val_folder = build_val_folder(**config.val)

    # generate train folder
    train_folder = build_train_folder(config.train)

    # update data files
    update_config_file(config)

    # init model
    model = YOLO(config.model.weights, cmd_args=config.model)
    model.train(
        data="data.yaml", epochs=config.model.epochs, batch=config.model.batch
    )
    wandb.finish()
    shutil.rmtree(val_folder, ignore_errors=True)
    shutil.rmtree(train_folder, ignore_errors=True)


def update_config_file(config):
    with open(config.model.data, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["path"] = os.getcwd()
    with open("data.yaml" , mode="w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    main()
