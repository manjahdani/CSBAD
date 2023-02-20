import os
from ._utils import *
import cv2

import sys
sys.path.append(os.getcwd() + '/strategy/')
from utils import copy_subsample, list_files_without_extensions

def build_frame_folder(video_path, output_folder, output_name='video', extension='png'):
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N_digits = len(str(frame_count))
    i = 0
    while cap.isOpened() and i < frame_count:
        ret, frame = cap.read()
        if not ret:
            raise IOError(f'Error while reading video file {video_path}')
        else:
            file_name = f'{output_folder}/{output_name}_{i:0{N_digits}}.{extension}'
            cv2.imwrite(file_name, frame)
        i += 1

#Old function to do the train/val split
#The input is a video
def build_train_val_folders(video_path, output_parent_folder, item_name='frame', extension='png', val_set_size=300,
                         min_n_frame=500):
    # build folder tree
    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)
    if not os.path.exists(f'{output_parent_folder}/val_split'):
        os.makedirs(f'{output_parent_folder}/val_split')
    if not os.path.exists(f'{output_parent_folder}/val_split/images'):
        os.makedirs(f'{output_parent_folder}/val_split/images')
    if not os.path.exists(f'{output_parent_folder}/val_split/labels'):
        os.makedirs(f'{output_parent_folder}/val_split/labels')
    if not os.path.exists(f'{output_parent_folder}/train_split'):
        os.makedirs(f'{output_parent_folder}/train_split')
    if not os.path.exists(f'{output_parent_folder}/train_split/images'):
        os.makedirs(f'{output_parent_folder}/train_split/images')
    if not os.path.exists(f'{output_parent_folder}/train_split/labels'):
        os.makedirs(f'{output_parent_folder}/train_split/labels')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < min_n_frame:
        raise VideoLengthException(
            f'Video length ({frame_count} frames) is too short ! (minimum {min_n_frame} frames required !)')
    N_digits = len(str(frame_count))
    i = 0
    while cap.isOpened() and i < frame_count:
        ret, frame = cap.read()
        if not ret:
            raise IOError(f'Error while reading video file {video_path}')
        else:
            if i < frame_count - val_set_size:
                file_name = f'{output_parent_folder}/train_split/images/{item_name}_{i:0{N_digits}}.{extension}'
                cv2.imwrite(file_name, frame)
            else:
                file_name = f'{output_parent_folder}/val_split/images/{item_name}_{i:0{N_digits}}.{extension}'
                cv2.imwrite(file_name, frame)
        i += 1
    print('Folder tree built with success !')

#New function to build the validation set 
#The validation set is composed of the 300 last frames of the bank
#The input is the bank set
def build_val_folder(bank_folder, labels_folder, extension, val_set_size=300, min_n_frame=500):
    validationSet = list_files_without_extensions(bank_folder +'/images', extension=extension)[-val_set_size::]

    outfolder = os.getcwd()
    if not os.path.exists(f'{outfolder}/val'):
        os.makedirs(f'{outfolder}/val')
    if not os.path.exists(f'{outfolder}/val/images'):
        os.makedirs(f'{outfolder}/val/images')
    if not os.path.exists(f'{outfolder}/val/labels'):
        os.makedirs(f'{outfolder}/val/labels')
    
    val_folder = outfolder + '/val/'
    copy_subsample(validationSet, bank_folder, val_folder,imgExtension=extension,labelsFolder=labels_folder)
    return val_folder