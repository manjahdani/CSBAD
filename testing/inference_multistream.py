import sys
import os
import csv
import cv2
import json
import argparse
from tabulate import tabulate
import time
import logging

import gc
import torch

if __name__ == '__main__':
    sys.path.append(os.path.join(sys.path[0], '..', "yolov8", "ultralytics"))
    from ultralytics import YOLO
elif __name__ == 'testing.inference_multistream':
    sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
    from ultralytics import YOLO


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DATA_YAML = os.path.join(BASE_PATH, 'data.yaml')

STRATEGIES = ['random','top_confidence_max', 'least_confidence_max','n-first']
METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95', 'fitness']

def get_runs_summary(weights_path, project, wandb_project_name):
    summary = os.path.join(os.path.abspath(weights_path), wandb_project_name + '.csv')
    with open(summary, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    summary_processed = {}
    for line in lines:
        try:
            run_summary = json.loads(line.split('"')[1].replace("'", '"'))
            run_name = line.strip('\n').split(',')[-1]
            summary_processed[run_name] = run_summary
        except Exception as e:
            print("ERROR, Fix this first --->", line)
            logging.exception(e)
            sys.exit()
        continue

    return summary_processed


def get_weights(weights_path, project):
    weights = os.listdir(weights_path)
    weights = [os.path.join(os.path.abspath(weights_path), w) for w in weights if w.endswith('.pt') and project in w]
    return weights


def build_run_info(weight, dataset_path, project, summary):
    run = weight.split('/')[-1]
    run_name = run.split('.')[1]
    for strategy in STRATEGIES:
        if strategy in run.split('WALT')[1]:
            break
    if run_name in summary:
        # '-'.join(run.split(project)[1].strip('-').split('_')[0].split('-')),
        #teacher = run_name.split('_')[-1]
        parts = run_name.split("_")
        return {
            'id': run.split('.')[0],
            # 'data-name': run.split(project)[1].strip('-').split('_')[0].split('-')[0],
            'data-name': parts[0].split("-")[1],
            'strategy': strategy,
            'epochs': summary[run_name]['_step'],
            'best/epoch': summary[run_name]['best/epoch'],
            'samples': int(parts[-1]),
            'data': parts[0].split("-")[1], # *run.split(project)[1].strip('-').split('_')[0].split('-')
            'teacher': "teacher"  # Add this line
        }

def main(weights_path, csv_path, dataset_path, project, wandb_project_name, base_data_yaml, task):

    weights = get_weights(weights_path, project)
    summary = get_runs_summary(weights_path, project, wandb_project_name)
    runs = []
    for weight in weights:
        info = build_run_info(weight, dataset_path, project, summary)
        if info:
            runs += [{**info, 'model': weight}]
        
    if not runs:
        print('No valid weights/data were found for testing')
        return []

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'data-name', 'strategy', 'teacher', 'epochs', 'best/epoch', 'samples', *METRICS, 'camera_tested_on'])

    for i, run in enumerate(runs):
        try:
            print(f'[{i+1}/{len(runs)}] Testing... : {run["id"]}')
            for camera_folder in os.listdir('/home/dani/data/WALT-challenge'):
                camera_path = os.path.join('/home/dani/data/WALT-challenge', camera_folder)
                if os.path.isdir(camera_path):
                    build_yaml_file(base_data_yaml, camera_path)
                    print(run['model'])
                    model = YOLO(run['model'])
                    results = model.val(data=TMP_DATA_YAML, task=task)

                    if len(results) == len(METRICS):
                        with open(csv_path, 'a+') as f:
                            writer = csv.writer(f)
                            writer.writerow([run['id'], run['data-name'], run['strategy'], run['teacher'], run['epochs'],
                                             run['best/epoch'], run['samples'], *list(results.values()), camera_folder])
                    else:
                        print('TESTING ERROR. NOT SAVING !')

                    print(f"Run {run['id']}:{run['data-name']} on {camera_folder} just finished being tested..............Done")
                    print('Sleeping and clearing memory')
                    time.sleep(1)
                    torch.cuda.empty_cache()
                    gc.collect()
        except AssertionError as e:
            print(e)
        except Exception as e:
            logging.exception(e)

    return runs

def build_yaml_file(base_file, dataset):
    print(base_file)
    print('dataset',dataset)
    lines_to_write = []
    with open(base_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'path:' in line:
                lines_to_write += [f'path : {dataset}\n']
            else:
                lines_to_write += [line]

    with open(TMP_DATA_YAML, 'w') as f:
        f.writelines(lines_to_write)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-w', '--weight_path', type=str, required=True,
                    help='The path to the weights to be evaluated')
    ap.add_argument('-d', '--dataset_path', type=str, required=True,
                    help='The path to the dataset folder, it should contain each camera in a separated folder')
    ap.add_argument('-p', '--project', type=str, required=True,
                    help='This is not the project name used in wandb, this is the dataset name used as prefix')
    ap.add_argument('-wp', '--wandb_project', type=str, required=False,
                    help='This is the project name used in wandb')
    ap.add_argument('-y', '--data_template', type=str, required=True,
                    help='Template yaml file for the dataset')
    ap.add_argument('-f', '--folder', type=str, required=False, default='test',
                    help='Set the folder to be used for testing: val or test')
    args = ap.parse_args()

    if not args.wandb_project:
        args.wandb_project = args.project

    
    main(args.weight_path, args.dataset_path, args.project, args.wandb_project, args.data_template, args.folder)


