import os
import random
from utils import *
import numpy as np

def strategy_n_first(image_folder_path: str, n: int = 300) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    path_list = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder_path)]
    
    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')
        
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'N first strategy !')
    path_list.sort()
    output_list = path_list[:n]
    return output_list


def strategy_random(image_folder_path: str, n: int = 300, seed: int = 42) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param seed: randomization seed for reproducibility
    :return output_list: a list containing the selected images path
    """
    path_list = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder_path)]
    
    if(n<=0):
        raise SamplingException(f'You must select a strictly positive number of frames to select')
        
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'random strategy !')
    path_list.sort()
    random.seed(seed)
    output_list = random.sample(path_list, n)
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
    indices = np.linspace(0,len(path_list)-1,n,dtype="int")
    output_list=[path_list[s] for s in indices]
    return output_list