from operator import sub
import os, os.path
from strategies import *
import argparse

def simulate_log_files (folder_path, strat_name, n_start, n_max, step):
    
    assert strat_name in ['n_first', 'fixed_interval'], 'The method ' + strat_name + ' is not supported ' #SUPPORTED METHODS
    tmp = range(n_max//step)
    n_frame = [n_start + step*interval for interval in tmp]
    bank_folder = os.path.join(folder_path, 'bank')
    bank_imgs_folder = os.path.join(bank_folder, 'images')

    for n in n_frame:
        subsample_names = []
        if strat_name == 'n_first':
            subsample_names = strategy_n_first(bank_imgs_folder, n)
        elif strat_name == 'fixed_interval':
            subsample_names = strategy_fixed_interval(bank_imgs_folder, n)
        name = 'n_frames-'+str(n)+'-strat_name-'+strat_name
        create_log_file(folder_path, name, subsample_names)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--folder_path', type=str, required=True,
                    help='The path to the camera folder, \
                    containing "bank", "train", "val", "test" folders')
    ap.add_argument('-s', '--strat_name', type=str, required=True,
                    help='The name of the subsample strategy')
    ap.add_argument('--n_start', type=int, required=True,
                    help='The name of the subsample strategy')
    ap.add_argument('--step', type=int, required=True,
                    help='The name of the subsample strategy')
    ap.add_argument('--n_max', type=int, required=True,
                    help='The seed for random strategy')
    args = ap.parse_args()

    simulate_log_files (args.folder_path, args.strat_name, args.n_start, args.n_max, args.step)

