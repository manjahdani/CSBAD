import argparse
import os
from subsampling import *
from utils import copy_subsample

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n_frames', type=int, default=300,
                    help='The number of frames to sample')
    ap.add_argument('-f', '--folder_path', type=str, required=True,
                    help='The path to the camera folder, \
                    containing "bank", "train", "val", "test" folders')
    ap.add_argument('-s', '--strat_name', type=str, required=True,
                    help='The name of the subsample strategy')
    ap.add_argument('--seed', type=int, required=False, default=42,
                    help='The seed for random strategy')

    args = ap.parse_args()

    assert args.strat_name in ['n_first', 'random', 'fixed_interval']

    bank_folder = os.path.join(args.folder_path, 'bank')
    bank_imgs_folder = os.path.join(bank_folder, 'images')
    train_folder = os.path.join(args.folder_path, 'train')

    subsample_names = []
    if args.strat_name == 'n_first':
        subsample_names = strategy_n_first(bank_imgs_folder, args.n_frames)
    elif args.strat_name == 'random':

        subsample_names = strategy_random(bank_imgs_folder, args.n_frames, args.seed)
    elif args.strat_name == 'fixed_interval':
        subsample_names = strategy_fixed_interval(bank_imgs_folder, args.n_frames)
    
    name_file = ''
    for a in vars(args):
        if(a!='folder_path'):
            name_file = name_file + a + '-' + str(vars(args)[a]) + '-'
    create_log_file(str(args.folder_path), name_file, subsample_names)
    copy_subsample(subsample_names, bank_folder, train_folder)