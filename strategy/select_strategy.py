import argparse
import os
from subsampling import *
from utils import copy_subsample
from hydra.utils import call


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n_frames', type=int, default=300,
                    help='The number of frames to sample')
    ap.add_argument('-f', '--folder_path', type=str, required=True,
                    help='The path to the camera folder, \
                    containing "bank", "train", "val", "test" folders')
    ap.add_argument('--img_extension', type=str, required=True,
                    help='The image extension')
    ap.add_argument('--labels_folder', type=str, required=True,
                    help='The bank folder of labels')
    ap.add_argument('-s', '--strat_name', type=str, required=True,
                    help='The name of the subsample strategy')
    ap.add_argument('--seed', type=int, required=False, default=42,
                    help='The seed for random strategy')
    ap.add_argument('--difference_ratio', type=float, required=False, default=4,
                    help='Optical flow difference ratio (Movement ratio)')
    ap.add_argument('--movement_percent', type=int, required=False, default=90,
                    help='Percentage of movement frames vs other frames')
    ap.add_argument('--entropy_file', type=str, required=False,
                    help='Path to the entropy file')
    ap.add_argument('--val_size', type=int, required=False, default=300, help='Validation set size. Default is 300')
    args = ap.parse_args()

    assert args.strat_name in ['n_first', 'random', 'fixed_interval', 'flow_diff', 'flow_interval_mix', 'entropy','frequency']

    bank_folder = os.path.join(args.folder_path, 'bank')
    bank_imgs_folder = os.path.join(bank_folder, 'images')
    train_folder = os.path.join(args.folder_path, 'train')
    difference_ratio = args.difference_ratio
    movement_percent = args.movement_percent

    subsample_names = []
    if args.strat_name == 'n_first':
        subsample_names = strategy_n_first(image_folder_path=bank_imgs_folder, n=args.n_frames, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'random':
        subsample_names = strategy_random(image_folder_path=bank_imgs_folder, n=args.n_frames, seed=args.seed, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'fixed_interval':
        subsample_names = strategy_fixed_interval(image_folder_path=bank_imgs_folder, n=args.n_frames, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'flow_diff':
        subsample_names = strategy_dense_optical_difference(image_folder_path=bank_imgs_folder, n=args.n_frames,
                                                            difference_ratio=difference_ratio, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'entropy':
        subsample_names = strategy_best_entropy(image_folder_path=bank_imgs_folder, entropy_file=args.entropy_file, n=args.n_frames, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'flow_interval_mix':
        subsample_names = strategy_flow_interval_mix(image_folder_path=bank_imgs_folder, n=args.n_frames,
                                                     difference_ratio=difference_ratio,
                                                     movement_percent=movement_percent, imgExtension=args.img_extension, val_size=args.val_size)
    elif args.strat_name == 'frequency':
        subsample_names = strategy_frequency(image_folder_path=bank_imgs_folder, n=args.n_frames, imgExtension=args.img_extension, bank_folder_path = bank_folder)

    name_file = ''
    for a in vars(args):
        if (a != 'folder_path' and a != 'entropy_file'):
            name_file = name_file + a + '-' + str(vars(args)[a]) + '-'

    create_log_file(str(args.folder_path), name_file, subsample_names)
    copy_subsample(subsample_names, bank_folder, train_folder,imgExtension=args.img_extension,labelsFolder=args.labels_folder)


def build_train_folder(config):
    subsample_names = call(config.strategy)
    copy_subsample(subsample_names, **config.subsample)



