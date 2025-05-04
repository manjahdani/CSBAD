import os
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np

from subsampling.dataset_builder import build_val_folder, build_train_folder

#sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
from ultralytics import RTDETR
from ultralytics import settings
settings.update({"wandb": True})

@hydra.main(version_base=None, config_path="experiments", config_name="experiment")
def train(config):
    # Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda:0"  # Use GPU
    else:
        device = None  # Use CPU

    config.cam_week_pairs, n_cameras = generate_cameras_pairs(config.dataset.name)

    if(config.training_mode=="cst_maturity"):
        if (config.N_streams != "null"):
            print(f"Mode: const maturity, base_epoch: {config.model.epochs}, streams: {config.N_streams}, new epoch: {int(config.model.epochs * config.N_streams / n_cameras)}")
            config.model.epochs = int(config.model.epochs*config.N_streams/n_cameras)
            config.epochs=config.model.epochs
        else:
            raise ValueError("For constant maturity study, 'N_streams' (total streams) is required.")


        

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

    # init model
    model = RTDETR(config.model.weights)

   # train model
    model.train(
        data="data.yaml",
        epochs=config.model.epochs,
        name=config.model.name,
        batch=config.model.batch,
        project=config.project,
        device=device,
        single_cls=True,
        pretrained=True,
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


def generate_cameras_pairs(input_string):
    # Find the index of the '-'
    dash_index = input_string.index('-')

    # Find the index of the '-week'
    week_index = input_string.index('-week')

    # Get the cameras and weeks as lists of characters
    cameras = input_string[dash_index+4:week_index].split('o')
    weeks = input_string[week_index+5:].split('o')

    # Check that the number of cameras is equal to the number of weeks
    if len(cameras) != len(weeks):
        raise ValueError("The number of cameras must be equal to the number of weeks")

    # Generate the output
    output = [{'cam': int(cam), 'week': int(week)} for cam, week in zip(cameras, weeks)]
    
    return output,len(cameras)


if __name__ == "__main__":
    train()
