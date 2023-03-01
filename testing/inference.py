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
elif __name__ == 'testing.inference':
    sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
    from ultralytics import YOLO


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DATA_YAML = os.path.join(BASE_PATH, 'data.yaml')

STRATEGIES = ['n_first', 'fixed_interval', 'flow_diff', 'flow_interval_mix', 'random', 'entropy', 'frequency',
               "topconfidence_max","topconfidence_sum","movement",
               "uniformStreambased", #Random stream-based
               "threshTopconfidenceStreambased_sum", "threshTopconfidenceStreambased_max", "threshLeastconfidenceStreambased_max", #stream-based
               "bernoulliLeastconfidenceStreambased_max","bernoulliTopconfidenceStreambased_max",
               "confidence_max", "confidence_min"]
# METRICS = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness']
METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95', 'fitness']

def get_runs_summary(weights_path, project, wandb_project_name):
    summary = os.path.join(os.path.abspath(weights_path), wandb_project_name + '.csv')
    with open(summary, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()[1:]
        # lines = [l.strip().split(',') for l in lines]

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
        if strategy in run.split(project)[1]:
            break
    if run_name in summary:
        # '-'.join(run.split(project)[1].strip('-').split('_')[0].split('-')),
        return {
            'id': run.split('.')[0],
            # 'data-name': run.split(project)[1].strip('-').split('_')[0].split('-')[0],
            'data-name': '-'.join(run.split(project)[1].strip('-').split('_')[0].split('-')),
            'strategy': strategy,
            'epochs': summary[run_name]['_step'],
            'best/epoch': summary[run_name]['best/epoch'],
            'samples': int(run.split(project)[1].split(strategy)[1].strip('_').split('.')[0].split('-')[0].split('_')[0]),
            'data': os.path.join(dataset_path, run.split(project)[1].strip('-').split('_')[0].split('-')[0]), # *run.split(project)[1].strip('-').split('_')[0].split('-')
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
        return

    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'data-name', 'strategy', 'epochs', 'best/epoch', 'samples', *METRICS])

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
    print(tabulate([r.values() for r in runs], headers=['RUN-ID', 'DATA-SHORT-NAME', 'STRATEGY', 'EPOCHS', 'BEST/EPOCH', 'SAMPLES', 'DATA', 'MODEL', 'TESTED']))
    
    # testing
    for i, run in enumerate(runs):
        try:
            if run['tested'] == 'YES':
                print(f"Run {run['id']}:{run['data-name']} has already been tested.............Done")
                continue
            
            print(f'[{i+1}/{len(runs)}] Testing... : {run["id"]}')
            build_yaml_file(base_data_yaml, run['data'])
            model = YOLO(run['model'])
            results = model.val(data=TMP_DATA_YAML, task=task)
            
            if len(results) == len(METRICS):
                with open(csv_path, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([run['id'], run['data-name'], run['strategy'], run['epochs'], run['best/epoch'], run['samples'], *list(results.values())])
            else:
                print('TESTING ERROR. NOT SAVING !')

            print(f"Run {run['id']}:{run['data-name']} just finished being tested..............Done")
            print('Seelping and clearing memory')
            time.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except AssertionError as e:
            print(e)
        except Exception as e:
            logging.exception(e)

    return runs



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
                    help='This is not the project name used in wandb, this is the dataset name used as prefix')
    ap.add_argument('-p', '--wandb_project', type=str, required=False,
                    help='This is the project name used in wandb')
    ap.add_argument('-y', '--data-template', type=str, required=True,
                    help='Template yaml file for the dataset')
    ap.add_argument('-f', '--folder', type=str, required=False, default='test',
                    help='Set the folder to be used for testing: val or test')
    args = ap.parse_args()

    if not args.wandb_project:
        args.wandb_project = args.project

    main(args.weights_path, args.csv_path, args.dataset_path, args.project, args.wandb_project, args.data_template, args.folder)
