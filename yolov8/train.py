import argparse

import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "yolov8", "ultralytics"))
from ultralytics import YOLO


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ## wandb
    ap.add_argument('-n', '--name', type=str, required=True,
                    help='name of the run, will sync to wandb')
    ap.add_argument('-e', '--entity', type=str, required=True,
                    help='wandb entity name (username or team name)')
    ap.add_argument('-p', '--project', type=str, required=True,
                    help='wandb project name')

    ## training
    ap.add_argument('-d', '--data', type=str, default = 'coco128.yaml', required=False,
                    help='path to data yaml file')
    ap.add_argument('-w', '--weights', type=str, default='yolov8n.pt', required=False,
                help='path to weights, default = yolov8n.pt')
    ap.add_argument('-b', '--batch', type=int, default=16, required=False,
                help='batch size, default = 16')
    ap.add_argument('-ep', '--epochs', type=int, default=10, required=False,
                help='epochs, default = 10')

    args = ap.parse_args()

    # assert len(args.name) and len(args.entity) and len(args.project), \
    #     'Not all wandb required parameters have been specified'

    model = YOLO(args.weights, cmd_args = args) # pass any model type
    model.train(data = args.data, epochs = args.epochs, batch = args.batch)
