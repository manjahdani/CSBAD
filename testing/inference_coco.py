import sys
import os
import csv
import argparse
from tabulate import tabulate
import time
import pandas as pd
import logging

import gc
import torch

from ultralytics import RTDETR


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DATA_YAML = os.path.join(BASE_PATH, 'data.yaml')

METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95', 'fitness']
MODELS = ['rtdetr-l', 'rtdetr-x']

def main(run_prefix, csv_path, datasets, base_data_yaml, task):
        # Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda:0"  # Use GPU
    else:
        device = None  # Use CPU
    torch.cuda.set_device(device)
    runs = []
    to_do = []
    for d in datasets:
        for m in MODELS:
            runs += [{'model': m, 'data-name': d.split('->')[0], 'data': d.split('->')[1]}]
            to_do += [True]#[model == m]

    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 
                            'source_dataset', 'source_domain','source_period',
                            'target_domain', #@FIXME Should include target_dataset and target_period
                            'student', 'teacher', 
                            'epochs',
                            'strategy',
                            'setting', 
                            'samples', 
                            *METRICS])
    testable = 0
    df=pd.read_csv(csv_path)


    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
        for run in runs:
            run['tested'] = 'NO'
            unique_strategies = df[df['source_domain'] == run['data-name']]['strategy'].unique()
            print (unique_strategies)
            if run['model'] in unique_strategies:
                    run['tested'] = 'YES'
                    testable += 1

    print(f'Found {len(runs)} runs of which {testable} need testing')
    print(tabulate([r.values() for r in runs], headers=['MODEL', 'DATA-SHORT-NAME', 'DATA', 'TESTED']))
    

    # testing
    for i, run in enumerate(runs):
        try:
            if not to_do[i]:
                continue

            if run['tested'] == 'YES':
                print(f"Run {run['model']}:{run['data-name']} has already been tested.............Done")
                continue

            print(f'[{i+1}/{len(runs)}] Testing... : {run["model"]} {run["data-name"]}')
            build_yaml_file(base_data_yaml, run['data'])
            
            model = RTDETR(model=run['model'] + '.pt')

            results = model.val(data=TMP_DATA_YAML, 
                                task=task, 
                                device=device, 
                                imgsz=640, 
                                conf=0.4, 
                                iou=0.7, 
                                batch=1, 
                                single_cls=True)
            
            results_dict= results.results_dict
            if len(results_dict) == len(METRICS):
                with open(csv_path, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow(['cocoDETR', 
                                     run_prefix, run['data-name'], "undefined",
                                     run['data-name'],
                                     run['model'], 
                                     "null", 
                                     0,
                                     "none","none",0, 
                                     *list(results_dict.values())])
            else:
                print('TESTING ERROR. NOT SAVING !')

            print(f"Run {run['model']}:{run['data-name']} just finished being tested..............Done")
            print('Seelping and clearing memory')
            time.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logging.exception(e)


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

    ap.add_argument('-m', '--model', type=str, required=True, default='rtdetr-l', 
                    choices = MODELS,
                    help='which pretrained model to use')
    ap.add_argument('-c', '--csv_path', type=str, required=False,
                    help='The path to the CSV with the results')
    ap.add_argument('-d', '--dataset', type=str, required=True, action='append',
                    help='Dataset to test, can specify multiple by using the flag multiple times, please use this format name->path')
    ap.add_argument('-y', '--data-template', type=str, required=True,
                    help='Template yaml file for the dataset')
    ap.add_argument('-f', '--folder', type=str, required=False, default='test',
                    help='Set the folder to be used for testing: val or test')
    args = ap.parse_args()

    main(args.model, args.csv_path, args.dataset, args.data_template, args.folder)
