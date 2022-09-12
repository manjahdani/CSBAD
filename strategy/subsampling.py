import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from utils import *

# Ignoring numpy warnings
import warnings
warnings.filterwarnings('ignore')

DEFAULT_SUB_SAMPLE = 300

def strategy_n_first(image_folder_path: str, n: int = DEFAULT_SUB_SAMPLE) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    path_list = list_files_without_extensions(image_folder_path)
    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'N first strategy !')
    path_list.sort()
    output_list = path_list[:n]
    return output_list


def strategy_random(image_folder_path: str, n: int = DEFAULT_SUB_SAMPLE, seed: int = 42) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param seed: randomization seed for reproducibility
    :return output_list: a list containing the selected images path
    """
    path_list = list_files_without_extensions(image_folder_path)
    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'random strategy !')
    path_list.sort()
    random.seed(seed)
    output_list = random.sample(path_list, n)
    output_list.sort()
    return output_list


def strategy_fixed_interval(image_folder_path: str, n: int = 1) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    
    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    
    path_list = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder_path)]
    
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f' fixed interval strategy !')
    path_list.sort()   
    step = len(path_list)//n   
    indices = np.arange(0,n)*step
    output_list=[path_list[s] for s in indices]
    return output_list


def strategy_dense_optical_difference(image_folder_path: str, n: int = DEFAULT_SUB_SAMPLE, jump = 2, fill_missing: bool = True,
            difference_ratio: float = 4., fixer_precision: float = 0.001) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param fill_missing: if the number of frames found is less than n, fill with random frames. 
    :param difference_ratio: ratio of changes in the frame
    :param fixer_precision: for fixing outliers (like the entire camera moving from wind). best choices are less or equal to 0.01
    :return output_list: a list containing the selected images path (check for list len if fill_missing is set to False)
    """

    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')

    path_list = list_files_without_extensions(image_folder_path)
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'random strategy !')
    path_list.sort()

    default_ext = '.png' 

    output_list = [0]
    frame1 = cv2.imread( os.path.join(image_folder_path, path_list[ output_list[-1] ] + default_ext) )
    mask = generate_mask(frame1)

    data_to_process = range(1, len(path_list), jump + 1)
    print("Running Optical Flow Difference Calcs :")
    for e, i in enumerate(data_to_process):
        frame2 = cv2.imread( os.path.join(image_folder_path, path_list[i] + default_ext) )
        result = optical_flow_compare(frame1, frame2, mask, fix_outliers = True, fixer_precision = fixer_precision)
        mag_sum = result.sum() / (1080*1920)
        
        if mag_sum > difference_ratio:
            output_list += [i]
            frame1 = cv2.imread( os.path.join(image_folder_path, path_list[ output_list[-1] ] + default_ext) )

        print(f"\rProcessed [{e + 1}/{len(data_to_process) + 1}] (Default jumping by {jump} frames), Selected [{len(output_list)}/{n}]", end = "")

        if len(output_list) >= n:
            output_list_processed = [path_list[i] for i in output_list]
            return output_list_processed[:n]

    output_list_processed = [path_list[i] for i in output_list]
    output_list_processed.sort()

    if len(output_list_processed) >= n:
        return output_list_processed[:n]
    elif fill_missing:
        print(f"Selected [{len(output_list)}/{n}] Through Movement Analysis (Optical Flow Difference), Will fill the rest with other frames...")
        output_list_processed = fill_in_between(path_list, output_list_processed, n)
    else:
        print(f"Selected [{len(output_list)}/{n}] Through Movement Analysis (Optical Flow Difference). Will not fill the rest since filling is disabled.")
        return output_list_processed


def strategy_flow_interval_mix(image_folder_path: str, n: int = DEFAULT_SUB_SAMPLE,
            movement_percent: int = 50, difference_ratio: float = 4.) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param fill_missing: if the number of frames found is less than n, fill with random frames. 
    :param difference_ratio: ratio of changes in the frame
    :param movement_percent: % of frames where movement was detect, the rest will be filled with other frames
    :return output_list: a list containing the selected images path
    """

    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')

    path_list = list_files_without_extensions(image_folder_path)
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'random strategy !')
    path_list.sort()
    movement_frames = strategy_dense_optical_difference(
        image_folder_path, 
        len(path_list),
        difference_ratio = difference_ratio,
        # These don't change
        jump = 2, 
        fill_missing = False,
    )
    movement_frames.sort()

    other_frames = list( set(path_list).difference(set(movement_frames)) )
    other_frames.sort()

    n_movement =  ((n * movement_percent) // 100)
    n_other = n - n_movement

    movement_indices = get_interval_indices(movement_frames,n_movement)
    other_indices = get_interval_indices(other_frames, n_other)

    output_list = \
        [ movement_frames[s] for s in movement_indices ] + \
        [ other_frames[s] for s in other_indices ]

    output_list.sort()
    return output_list