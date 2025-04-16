import argparse
import csv
import gc
import json
import logging
import os
import pandas as pd
import sys
import time
import torch
import re
if __name__ == '__main__':
    sys.path.append(os.path.join(sys.path[0], '..', "yolov8", "ultralytics"))
    from ultralytics import YOLO
elif __name__ == 'testing.inference':
    sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
    from ultralytics import YOLO

# NAMING CONVENTION:
# {Source_domain}_{Target_domain}_{Student}_{Teacher}_{Strategy}_{Active-Learning-Setting}_{Total_Samples} # to add _{Iteration_Level}_{Epochs}_{Validation_Set}
# Finer-granularity can be expected as
# {Source_domain} = {dataset}-{domain}-{period}
#
# The use of _ is not allowed in any of the subcomponents. E.g. Strategy = 'least_confidence_max' is wrong. Instead, one should have 'least-confidence-max'
#
# In the case of multi-run, we assume a balanced class therefore there is only need to have the name of the domain and their respective period as
# domain = domain{1,2,3,n}; E.g., if we trained on cam 1,11,120 and used the 1st week for cam 1, the second and fourth week for cam 11 and 120 then it should be "cam1,11,120", "week1,2,4".
#
# run = 2kq3iot2.WALT-cam3,4-week5,6_yolov8n_yolov8x6_least-confidence-max_stream-based_16
# run_name = WALT-cam3,4-week5,6_yolov8n_yolov8x6_least-confidence-max_stream-based_16

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DATA_YAML = os.path.join(BASE_PATH, 'data.yaml')

METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95', 'fitness']

def get_runs_summary(weights_path, wandb_project_name):
    summary = os.path.join(os.path.abspath(weights_path), wandb_project_name + '.csv')
    with open(summary, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    summary_processed = {}
    for line in lines:
        try:
            run_summary = json.loads(line.split('"')[1].replace("'", '"'))
            run_name = line.strip('\n').split(',')[-1]
            # Search for 'epochs' in the config part of the line
            match = re.search(r"'epochs': (\d+)", line)
            if match:
                epochs = int(match.group(1))
            else:
                # Default to None or an appropriate value if 'epochs' not found
                epochs = None
            # Add 'epochs_asked' to run_summary
            run_summary['epochs_asked'] = epochs
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
    if run_name in summary:
        parts = run_name.split("_")
        
        return {
            'id': run.split('.')[0],
            'data-name': run.split(project)[1].strip('-').split('_')[0].split('-')[0],
            'source_dataset':parts[0].split("-")[0],
            'source_domain': parts[0].split("-")[1],
            'source_period': parts[0].split("-")[2],
            'student' : parts[1],
            'teacher':  parts[2],
            'strategy': parts[3],
            'epochs': summary[run_name]['_step'],
            'setting':parts[4],
            'data':os.path.join(dataset_path, parts[0].split("-")[1]), #test_set path
            'samples': int(parts[5]),
        }

def main(weights_path, csv_path, dataset_path, project, wandb_project_name, base_data_yaml, task, target_domains, self_only=False):
    #Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda:0"  # Use GPU
    else:
        device = None  # Use CPU
    torch.cuda.set_device(device)
    weights = get_weights(weights_path, project)
    summary = get_runs_summary(weights_path, wandb_project_name)
    runs = []
    for weight in weights:
        info = build_run_info(weight, dataset_path, project, summary)
        if info:
            runs += [{**info, 'model': weight}]
        
    if not runs:
        print('No valid weights/data were found for testing')
        return []
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
    df=pd.read_csv(csv_path)


    print(f'Found {len(runs)} runs for project {project} need testing')
   
    for i, run in enumerate(runs):
        try:
            print(f'[{i+1}/{len(runs)}] Testing... : {run["id"]}')
            
            if self_only:
                print(f'Only testing on camera', run['source_domain'])
                target_domains=[run['source_domain']]
            for target_domain in target_domains:
                if run['id'] in df['run_id'].values:
                    filtered_df = df[df['run_id'] == run['id']]
                    if target_domain in filtered_df['target_domain'].values:
                        print(f"Run {run['id']}:{run['data-name']} has already been tested on {target_domain}.............Done")
                        continue
                    
                target_domain_path = os.path.join(dataset_path,target_domain)
                build_yaml_file(base_data_yaml, target_domain_path)
                model = YOLO(run['model'])
                results = model.val(data=TMP_DATA_YAML, task=task, device=device)
                results_dict= results.results_dict
                if len(results_dict) == len(METRICS):
                    with open(csv_path, 'a+') as f:
                        writer = csv.writer(f)
                        writer.writerow([run['id'], 
                                            run['source_dataset'], run['source_domain'], run['source_period'], 
                                            target_domain, 
                                            run['student'],run['teacher'], 
                                            run['epochs'],
                                            run['strategy'],
                                            run['setting'], 
                                            run['samples'], 
                                            *list(results_dict.values())])
                else:
                    print('TESTING ERROR. NOT SAVING !')

                print(f"Run {run['id']}:{run['data-name']} on target domain {target_domain} just finished being tested..............Done")
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
    ap.add_argument('-td', '--target_domains', type=str, required=True,
                    help='Set the target_domains to be used for testing')
    ap.add_argument('-td', '--self_only', type=bool, required=False, default=False,
                    help='Set to test only')
    args = ap.parse_args()

    if not args.wandb_project:
        args.wandb_project = args.project

    
    main(args.weight_path, args.dataset_path, args.project, args.wandb_project, args.data_template, args.folder, args.target_domains.split(','),args.self_only)


