import os
import sys
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np
import subprocess
from subsampling.dataset_builder import build_val_folder, build_train_folder
from build_dataset import yolo_to_coco_oneclass
from hydra.utils import to_absolute_path
import uuid
from pprint import pprint


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
    path_run = update_config_file(config)
    print(path_run)
    train_path_json_labels,_ = yolo_to_coco_oneclass(
        os.path.join(path_run, "train", "labels"),
        os.path.join(path_run, "train", "images")
    )
    val_path_json_labels, _ = yolo_to_coco_oneclass(
        os.path.join(path_run, "val", "labels"),
        os.path.join(path_run, "val", "images")
    )

    # Copy and adapt NanoDet config
    orig_cfg_path = to_absolute_path("/home/dani/CSBAD/nanodet/config/nanodet-plus-m_416.yml")
    os.makedirs("/home/dani/CSBAD/workspace/", exist_ok=True)
    
    run_name = f"{str(uuid.uuid4())[:8]}_{config.model.name}_{config.model.epochs}"
    workspace_dir = os.path.join("/home/dani/CSBAD/workspace/", run_name)
    
    os.makedirs(workspace_dir, exist_ok=True)
    custom_cfg_path = os.path.join(workspace_dir, "config.yml")
    shutil.copy(orig_cfg_path, custom_cfg_path)

    with open(custom_cfg_path, 'r') as f:
        nanodet_cfg = yaml.safe_load(f)
    
    pprint(nanodet_cfg['model'])
    
    # Modify the necessary fields
    nanodet_cfg ['model']['arch']['head']['num_classes'] = 1
    nanodet_cfg ['model']['arch']['aux_head']['num_classes'] = 1
    nanodet_cfg ['class_names'] = ['vehicle']

    nanodet_cfg['data']['train']['img_path'] = os.path.join(path_run, "train", "images")
    nanodet_cfg['data']['train']['ann_path'] = train_path_json_labels
    nanodet_cfg['data']['val']['img_path'] = os.path.join(path_run, "val", "images")
    nanodet_cfg['data']['val']['ann_path'] = val_path_json_labels
    
    nanodet_cfg['save_dir'] = f"{workspace_dir}"
    nanodet_cfg['schedule']['total_epochs'] = str(config.model.epochs)
    nanodet_cfg['device']['precision'] = 16  # or 32

    #nanodet_cfg['schedule']['load_model'] = "/home/dani/CSBAD/checkpoints/nanodet-plus-m_416_checkpoint.ckpt"
    # init model
    with open(custom_cfg_path, 'w') as f:
        yaml.dump(nanodet_cfg, f)

    subprocess.run([
        "python",
        "/home/dani/CSBAD/nanodet/tools/train.py",
        custom_cfg_path
    ])

    # finish the run and remove tmp folders
    #wandb.finish()
    #shutil.rmtree(val_folder, ignore_errors=True)
    #shutil.rmtree(train_folder, ignore_errors=True)


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
    return data["path"]


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