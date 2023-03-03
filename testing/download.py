# Written on top of local package wandb==0.12.1 and remote api 0.13.3
# "wandb login" in a terminal before using this script
# Runs states : finished, running, crashed
# Important run params/functions : .id, .state, .files, .config, .summary, .history(), .file('filename.ext').download(), api.run(f"{ENTITY}/{PROJECT}/{run.id}")
import os
import sys
import argparse
import wandb
from tabulate import tabulate
from tqdm import tqdm

import pandas as pd 

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'models')

class Downloader:
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self.api = wandb.Api(timeout=25)
        self.runs_url = f'https://wandb.ai/{entity}/{project}/runs/'


    def get_runs(self, query_filter = None):
        finished, running, other = [], [], []

        print("Loading Runs DB...")
        if query_filter:
            finished_query = self.api.runs(f"{self.entity}/{self.project}", {
                "display_name": {"$regex": f".*{query_filter}.*"}
            }, per_page = 4)
        else:
            finished_query = self.api.runs(f"{self.entity}/{self.project}", per_page = 4)
        finished_query[0]
        l = finished_query.length
        
        print("Loading Runs to memory...")
        for i in tqdm(range(l)):
            run = finished_query[i]
            if run.state == 'finished':
                finished += [run]
            elif run.state == 'running':
                running += [run]
            else:
                other += [run]
        return finished, running, other


    def check_runs(self, runs, query_filter):
        print('Checking list of finished runs...')
        finished, running, other = self.get_runs(query_filter)
        for run in finished:
            if run.id in runs:
                runs[runs.index(run.id)] = run
            if run.name in runs:
                runs[runs.index(run.name)] = run

        runs_found = []
        for run in runs:
            if type(run).__name__ == 'str':
                print(f'Could not find {run}')
            else:
                runs_found += [run]
        return runs_found


    def download_model(self, run, download_dir = DEFAULT_DOWNLOAD_DIR):
        download_location = os.path.join(download_dir, f'{run.id}.{run.name}.pt')
        if os.path.exists(download_location):
            print(f"Skipping : '{run.id}.{run.name}.pt' already exists")
        else:
            try:
                model_artifact = self.api.artifact(f'{self.entity}/{self.project}/run_{run.id}_model:best')
                model_artifact.download(download_dir)
                if os.path.exists(os.path.join(download_dir, 'best.pt')):
                    os.rename(os.path.join(download_dir, 'best.pt'), download_location)
                else:
                    print('Skipping : Could not download.',end='\n\n')
            except Exception as e:
                print(f'Could not download {run} :', e)


    def save_summary(self, runs, download_dir = DEFAULT_DOWNLOAD_DIR, project = 'results'):
        summary_list, config_list, name_list = [], [], []
        for run in runs: 
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k,v in run.config.items()
                if not k.startswith('_')})
            
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })

        print('Saving WANDB summary to:', os.path.join(download_dir, f'{project}.csv'))
        runs_df.to_csv(os.path.join(download_dir, f'{project}.csv'))



def main(args):
    d = Downloader(args.entity, args.project)
    finished, running, other = d.get_runs(args.query_filter)

    runs_to_process = []
    if args.runs:
        runs = d.check_runs(args.runs, args.query_filter)
        runs_to_process = runs
    elif args.list_all:
        for run in finished + running + other:
            runs_to_process += [run]
    elif args.list_finished:
        runs_to_process = finished
    elif args.list_running:
        runs_to_process = running

    downloaded_nb = 0    
    runs_to_print = []
    runs_to_download = []
    for run in runs_to_process:
        download_location = os.path.join(args.folder, f'{run.id}.{run.name}.pt')
        downloaded = True 
        if os.path.exists(download_location):
            downloaded = 'Yes'
            downloaded_nb += 1
        else:
            downloaded = 'No'
            runs_to_download += [run]

        runs_to_print += [[run.id, run.name, run.state, downloaded, d.runs_url + run.id]]
        
    print(tabulate(runs_to_print, headers=['ID', 'NAME', 'STATUS', 'DOWNLOADED', 'URL']))

    if args.list_all:
        print(f'Models: {len(finished + running + other)}, Downloaded: {downloaded_nb} (In : {args.folder}). Can not download {len(running)}, still running.')
    elif args.list_finished or args.runs:
        print(f'Models: {len(finished)}, Downloaded: {downloaded_nb} (In : {args.folder})')
    elif args.list_running:
        print(f'Models: {len(running)}. Still running. Not able to download weights yet.')

    if args.download and runs_to_download:
        print('-------------------------')
        for run in runs_to_download:
            print(f'Downloading : {run.id}.{run.name}.pt')
            d.download_model(run, args.folder)

    try:
        print('Saving Runs summary !')
        d.save_summary(runs_to_process, args.folder, args.project)
    except Exception as e:
        print('Could not save summary:', e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-la', '--list-all', action = 'store_true', required = False, default = False, help='list all runs')
    ap.add_argument('-lf', '--list-finished', action = 'store_true', required = False, default = False, help='list finished runs')
    ap.add_argument('-lr', '--list-running', action = 'store_true', required = False, default = False, help='list running runs')
    ap.add_argument('-r', '--runs', nargs='+', required = False, help = 'Use a custom list of runs')
    ap.add_argument('-s', '--sort', action = 'store_true', required = False, default = False, help='sort listed')
    ap.add_argument('-f', '--folder', type = str, required = False, default = './models', help='Folder to download & check for local runs. use with one of the listing arguments to download')
    ap.add_argument('-d', '--download', action = 'store_true', required = False, default = False, help='Download listed models')
    ap.add_argument('-e', '--entity', type = str, required = True, help='Wandb Entity')
    ap.add_argument('-p', '--project', type = str, required = True, help='Wandb Project')
    ap.add_argument('-q', '--query_filter', type = str, required = False, default = None, help='Filter by strings in run names')
    args = ap.parse_args()

    if args.folder:
        args.folder = os.path.abspath(args.folder)

    if len(sys.argv) < 2:
        ap.print_help()
        sys.exit(1)
    else:
        main(args)
