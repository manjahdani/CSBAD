import os, os.path
import shutil
import glob 
import shutil
import numpy as np
import cv2
from typing import List, Tuple
from logging import warning

class SamplingException(Exception):
    pass

def copy_subsample(index,in_folder,out_folder,imgExtension,labelsFolder):
    """
    :param index: an array of the name of the images that are selected ('e.g. ['frame_0001','frame_0020'])
    :param in_folder: path to the directory of the source folder containing images and labels subfolders (e.g., "C:/banks")
    :param out_folder: path to the dest (e.g., "C:/train")

    Create a new directory that copies all the images and the labels following the index in a new folder 
    """
    print(labelsFolder)
    files = glob.glob(f'{out_folder}/*/*')
    if(len(files)>0):
        warning("Train folder was flushed. All files were removed")
        for f in files:
            os.remove(f)
    
    images = os.listdir(os.path.join(in_folder,"images")) #Source of the bank images
    labels = os.listdir(os.path.join(in_folder,labelsFolder)) # Source of the bank of labels
    
    os.makedirs(os.path.join(out_folder,"images"), exist_ok=True) #Create image directory in out_folder if it doesn't exist in out_folder
    os.makedirs(os.path.join(out_folder,"labels"), exist_ok=True) #Create labels directory in out_folder if it doesn't exist in out_folder
    
    for img in index:
        img_with_extension = img + str(".") + imgExtension
        img_with_label = img + ".txt"
        assert(img_with_extension in images),'Source bank folder does not contain image with name file - ' + img_with_extension
        assert(img_with_label in labels),'Source folder does not contain a file - ' + img_with_label
        shutil.copy(os.path.join(in_folder,"images",img_with_extension),os.path.join(out_folder,"images",img_with_extension))
        shutil.copy(os.path.join(in_folder,labelsFolder,img_with_label),os.path.join(out_folder,"labels",img_with_label))




def create_log_file(folder_path,name,frames_list):
    output_folder = os.path.dirname(folder_path) #EXAMPLE IF FOLDER PATH 'E:/dataset/S05C016', it gives the "E:/dataset"
    video_name = os.path.basename(folder_path)  #EXAMPLE IF FOLDER PATH 'E:/dataset/S05C016', it gives the "S05C016
    with open(os.path.join(output_folder, video_name + '-' + name + ".txt"), "w") as log:
        for f in frames_list:
            log.write(video_name + '/bank/' + f + ".png\n")


def list_files_without_extensions(path: str, extension: str = 'png') -> list:
    """
    :param path: path to scan for files 
    :param extensions: what type of files to scan for
    :return path_list: list of file names without the extensions
    """
    path_list = [os.path.splitext(filename)[0] for filename in os.listdir(path) if filename.endswith(extension)]
    return path_list


def generate_mask(frame: np.array, fill: int = 255) -> np.array:
    """
    :param frame: image
    :param fill: integer value to fill the matrix
    :return mask: matrix of the same shape as frame, containing the fill value
    """
    mask = np.zeros_like(frame)
    mask[..., 1] = fill
    return mask


def mat_clipper(mat: np.array) -> Tuple[np.array, float]:
    """
    :param mat: a numpy array
    :return normalized_mat: array of same dimensions as input mat, with very small repetitive values removed
    :return difference: difference between the mean of the input mat & output mat
    """
    mat = np.nan_to_num(mat, copy = True, nan = 0.0)
    mean = np.mean(mat)
    normalized_mat = mat - mean
    clipped_mat = normalized_mat.clip(min = 0) # clipping negative values
    difference = mean - np.mean(clipped_mat) # difference between old and new mean
    return clipped_mat, difference


def fill_in_between(full_array: list, filtered_array: list, length_required: int) -> list:
    """
    :param full_array: example [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]
    :param filtered_array: example [1, 3, 6, 8]
    :param length_required: 6
    :return padded_filtered_array: [1, 2, 3, 4, 6, 8]
    """
    
    full_array.sort()
    filtered_array.sort()


    while True:
        # Done !
        if len(filtered_array) == length_required: 
            break
        
        padded_filtered_array = []

        for i in range(len(filtered_array)):
            if filtered_array[i] not in padded_filtered_array:
                padded_filtered_array += [ filtered_array[i] ]
            
            # Done !
            if len(padded_filtered_array) == length_required: 
                break
            # Padding sufficient, Done !
            elif len(padded_filtered_array + filtered_array[i + 1:]) == length_required:
                padded_filtered_array += filtered_array[i + 1:]
                break

            # element already chosen by strategy, skip
            index_in_full_list = full_array.index(filtered_array[i])
            if index_in_full_list == len(full_array) - 1:
                continue

            if full_array[index_in_full_list + 1] in padded_filtered_array:
                continue
            elif full_array[index_in_full_list + 1] not in padded_filtered_array:
                padded_filtered_array += [ full_array[index_in_full_list + 1] ]
                # Done !
                if len(padded_filtered_array) == length_required: 
                    break
                # Padding sufficient, Done !
                elif len(padded_filtered_array + filtered_array[i + 2:]) == length_required:
                    padded_filtered_array += filtered_array[i + 2:]
                    break

        padded_filtered_array.sort()
        filtered_array = padded_filtered_array.copy()
        
    return filtered_array


def optical_flow_compare(frame1: np.array, frame2: np.array, mask: np.array, 
            fix_outliers: bool = True, fixer_precision: float = 0.001) -> np.array:
    """
    :param frame1: image
    :param frame2: image
    :param mask: a numpy array of the same dimension as image containing a duplicate value (0 to 255)
    :param fix_outliers: fixing outliers (like the entire camera moving from wind)
    :param fixer_precision: best choices are less or equal to 0.01
    :return rgb: image showing dense optical flow between two compared frames
    """

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 15, 15, 3, 5, 1.2, 0)
    
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # removing outliers
    if fix_outliers:
        old_difference = 1
        while True:
            magnitude, new_difference = mat_clipper(magnitude)
            if old_difference - new_difference < fixer_precision:
                break
            else:
                old_difference = new_difference

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # HSV TO RGB
    return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)


def get_interval_indices(samples: list, sub_samples_length: int) -> List[int]:
    step = len(samples) // sub_samples_length
    indices = np.arange(0, sub_samples_length) * step
    return indices