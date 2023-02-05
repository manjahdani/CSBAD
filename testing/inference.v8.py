import sys
import os
import csv
import cv2
import argparse
from tabulate import tabulate
import time

import gc
import torch

sys.path.append(os.path.join(sys.path[0], '..', "yolov8", "ultralytics"))
from ultralytics import YOLO

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DATA_YAML = os.path.join(BASE_PATH, 'data.yaml')

METRICS = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness']

def get_weights(weights_path, project):
    weights = os.listdir(weights_path)
    weights = [os.path.join(os.path.abspath(weights_path), w) for w in weights if w.endswith('.pt') and project in w]
    return weights

def build_run_info(weight, dataset_path, project):
    run = weight.split('/')[-1]
    return {
        'id': run.split('.')[0],
        'data': os.path.join(dataset_path, *run.split(project)[1].strip('-').split('_')[0].split('-')),
        'data-name': '-'.join(run.split(project)[1].strip('-').split('_')[0].split('-'))
    }

def main(weights_path, csv_path, dataset_path, project, base_data_yaml, task):
    weights = get_weights(weights_path, project)

    runs = []
    for weight in weights:
        info = build_run_info(weight, dataset_path, project)
        if info:
            runs += [{**info, 'model': weight}]
        
    if not runs:
        print('No valid weights/data were found for testing')
        return

    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'data-name', *METRICS])

    testable = 0
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
        for run in runs:
            run['tested'] = 'NO'
            for line in lines:
                if run['id'] in line:
                    run['tested'] = 'YES'
                    testable += 1

    print(f'Found {len(runs)} runs for project {project} of which {testable} need testing')
    print(tabulate([r.values() for r in runs], headers=['RUN-ID', 'DATA', 'MODEL', 'TESTED']))
    
    # testing
    for run in runs:
        try:
            if run['tested'] == 'YES':
                print(f"Run {run['id']}:{run['data-name']} has already been tested.............Done")
                continue
            
            print(f'Testing... : {run["id"]}')
            build_yaml_file(base_data_yaml, run['data'])
            model = YOLO(run['model'])
            results = model.val(data=TMP_DATA_YAML, task=task)
            
            if len(results) == len(METRICS):
                with open(csv_path, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([run['id'], run['data-name'], *list(results.values())])
            else:
                print('TESTING ERROR. NOT SAVING !')

            print(f"Run {run['id']}:{run['data-name']} just finished being tested..............Done")
            print('Seelping and clearing memory')
            time.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(e)



def build_yaml_file(base_file, dataset):
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

    ap.add_argument('-w', '--weights_path', type=str, required=True,
                    help='The path to the weights to be evaluated')
    ap.add_argument('-c', '--csv_path', type=str, required=False,
                    help='The path to the CSV with the results')
    ap.add_argument('-d', '--dataset_path', type=str, required=True,
                    help='The path to the dataset folder, it should contain each camera in a separated folder')
    ap.add_argument('-p', '--project', type=str, required=True,
                    help='The project name used in the wandb runs')
    ap.add_argument('-y', '--data-template', type=str, required=True,
                    help='Template yaml file for the dataset')
    ap.add_argument('-f', '--folder', type=str, required=True, default='test',
                    help='Template yaml file for the dataset')
    args = ap.parse_args()

    main(args.weights_path, args.csv_path, args.dataset_path, args.project, args.data_template, args.folder)