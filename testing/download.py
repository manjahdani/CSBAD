# Written on top of local package wandb==0.12.1 and remote api 0.13.3
# "wandb login" in a terminal before using this script
# Runs states : finished, running, crashed
# Important run params/functions : .id, .state, .files, .config, .summary, .history(), .file('filename.ext').download(), api.run(f"{ENTITY}/{PROJECT}/{run.id}")
import os
import sys
import argparse
import wandb
from tabulate import tabulate


DEFAULT_DOWNLOAD_DIR = './models'

class Downloader:
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self.api = wandb.Api()
        self.runs_url = f'https://wandb.ai/{entity}/{project}/runs/'


    def get_runs(self):
        runs = self.api.runs(f"{self.entity}/{self.project}")
        finished = [run for run in runs if run.state == 'finished']
        running = [run for run in runs if run.state == 'running']
        other = [run for run in runs if run.state not in ['finished', 'running']]
        return finished, running, other


    def check_runs(self, runs):
        print('Checking list of finished runs...')
        finished, running, other = self.get_runs()
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
            model_artifact = self.api.artifact(f'{self.entity}/{self.project}/run_{run.id}_model:best')
            model_artifact.download(download_dir)
            if os.path.exists(os.path.join(download_dir, 'best.pt')):
                os.rename(os.path.join(download_dir, 'best.pt'), download_location)
            else:
                print('Skipping : Could not download.',end='\n\n')


def main(args):
    d = Downloader('trail22kd', 'kd')
    finished, running, other = d.get_runs()

    runs_to_process = []
    if args.runs:
        runs = d.check_runs(args.runs)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-la', '--list-all', action = 'store_true', required = False, default = False, help='list all runs')
    ap.add_argument('-lf', '--list-finished', action = 'store_true', required = False, default = False, help='list finished runs')
    ap.add_argument('-lr', '--list-running', action = 'store_true', required = False, default = False, help='list running runs')
    ap.add_argument('-r', '--runs', nargs='+', required = False, help = 'Use a custom list of runs')
    ap.add_argument('-s', '--sort', action = 'store_true', required = False, default = False, help='sort listed')
    ap.add_argument('-f', '--folder', type = str, required = False, default = './models', help='Folder to download & check for local runs. use with one of the listing arguments to download')
    ap.add_argument('-d', '--download', action = 'store_true', required = False, default = False, help='Download listed models')
    args = ap.parse_args()

    if len(sys.argv) < 2:
        ap.print_help()
        sys.exit(1)
    else:
        main(args)