import os
import sys
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np

from subsampling.dataset_builder import build_val_folder, build_train_folder

sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
from ultralytics import YOLO


@hydra.main(version_base=None, config_path="experiments", config_name="experiment")
def train(config):
    # Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda:0"  # Use GPU
    else:
        device = None  # Use CPU

    
    cam_week_pairs = config.cam_week_pairs

    cams = [str(pair['cam']) for pair in cam_week_pairs]
    weeks = [str(pair['week']) for pair in cam_week_pairs]

    config.dataset.name = f"{config.dataset.basename}-cam{'e'.join(cams)}-week{'e'.join(weeks)}"

    config.model.name=config.dataset.name +f"_{config.student}"+f"_{config.teacher}"+f"_{config.train.strategy.name}" 
    
    # Set the default device for tensors
    torch.cuda.set_device(device)
    # fix the seed
    set_random(config.seed)

    # generate validation folder
    val_folder = build_val_folder(**config.val)

    # generate train folder
    train_folder = build_train_folder(config.train)

    # update data files
    update_config_file(config)
    print(config.model.weights)
    # init model
    model = YOLO(config.model.weights, cmd_args=config.model)
    # train model
    model.train(
        data="data.yaml",
        epochs=config.model.epochs,
        batch=config.model.batch,
        device=device,
    )

    # finish the run and remove tmp folders
    wandb.finish()
    shutil.rmtree(val_folder, ignore_errors=True)
    shutil.rmtree(train_folder, ignore_errors=True)


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def update_config_file(config):
    with open(config.model.data, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["path"] = os.getcwd()
    with open("data.yaml", mode="w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    train()
