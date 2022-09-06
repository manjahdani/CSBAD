from dataset import *
import argparse

'''
WARNING : should be fixed in order to match the correct dataset folder tree with /bank /train /val /test
'''

parser = argparse.ArgumentParser()
parser.add_argument('--vp', type=str, required=True, help='Path to the video file')
parser.add_argument('--out', type=str, required=True, help='Output parent folder to build the tree. Created if '
                                                           'does not exist')
parser.add_argument('--n', type=str, required=False, default='frame', help='Item name. Default is frame')
parser.add_argument('--e', type=str, required=False, help='Image extension. Default is png')
parser.add_argument('--val_size', type=int, required=False, default=300, help='Validation set size. Default is 300')
parser.add_argument('--mf', type=str, required=False, default=500, help='Minimum number of frames in the video. '
                                                                        'Default is 500')
args = parser.parse_args()

video_path = args.vp
output_parent_folder = args.out


build_train_val_folders(video_path, output_parent_folder, item_name='frame', extension='png', val_set_size=args.val_size,
                        min_n_frame=args.mf)
