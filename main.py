import os
import sys
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np

from strategy.subsample import build_val_folder, build_train_folder

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

    make_training(config)


def update_config_file(data_file_path):
    with open(data_file_path, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["path"] = os.getcwd()
    with open("data.yaml" , mode="w") as f:
        yaml.dump(data, f)
    
def make_training(sub_config):

    # generate validation folder
    val_folder = build_val_folder(**sub_config.val)

    # generate train folder
    train_folder = build_train_folder(**sub_config.train)

    # update data files
    update_config_file(sub_config.model.data)

    # init model
    model = YOLO(sub_config.model.weights, cmd_args=sub_config.model)
    model.train(
        data="data.yaml", epochs=sub_config.model.epochs, batch=sub_config.model.batch
    )
    wandb.finish()
    shutil.rmtree(val_folder, ignore_errors=True)
    shutil.rmtree(train_folder, ignore_errors=True)

if __name__ == "__main__":
    main()
