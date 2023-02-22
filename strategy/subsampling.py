import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from utils import *
import pandas as pd
from os.path import exists
from frequency_utils import *
# Ignoring numpy warnings
import warnings
warnings.filterwarnings('ignore')

DEFAULT_SUB_SAMPLE = 300

def strategy_least_confidence(image_labels_path: str, n: int = DEFAULT_SUB_SAMPLE, aggregation_function: str = "max",
                              **kwargs) -> list:
    """
    Performs active learning for object detection using the confidence scores.

    Parameters:
    - image_labels_path: paths to the .txt files with the object detections (last element of each line = confidence score).
    - n: number of images to label.
    - aggregation_function: how to compute the confidence of an image based on the confidence of the single objects:
        a) "max": minmax approach, where the confidence of an image is given by the most confidently detected object.
        b) "min": confidence of the whole image is given by the most difficult object detected.
        c) "mean": average of all the confidence scores, it is not sensible to the number of objects detected.
        d) "sum": sensible to the number of objects detected.

    Returns:
    - images_to_label: list of strings, names of the images to be labeled (without extension)
    """
    txt_files = [filename for filename in os.listdir(image_labels_path)]
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(txt_files):
        raise SamplingException(f'Image bank contains {len(txt_files)} frames, but {n} frames where required for the '
                                f'least confidence strategy !')
    confidences = []
    for txt_file in txt_files:
        with open(os.path.join(image_labels_path, txt_file), 'r') as f:
            lines = f.readlines()
            if lines:
                # If the file is not empty, compute the image confidence score
                if aggregation_function == 'max':
                    image_confidence = max([float(line.strip().split()[-1]) for line in lines])
                elif aggregation_function == 'min':
                    image_confidence = min([float(line.strip().split()[-1]) for line in lines])
                elif aggregation_function == 'mean':
                    object_confidences_scores = [float(line.strip().split()[-1]) for line in lines]
                    image_confidence = sum(object_confidences_scores) / len(object_confidences_scores)
                elif aggregation_function == 'sum':
                    object_confidences_scores = [float(line.strip().split()[-1]) for line in lines]
                    image_confidence = sum(object_confidences_scores)
                else:
                    raise SamplingException(f'You must select a valid aggregation function')
                confidences.append((txt_file, image_confidence))

    # Sort the images based on the confidence
    confidences = sorted(confidences, key=lambda x: x[1])

    # Get the paths to the images with the lowest confidence
    images_to_label = [os.path.splitext(img)[0] for img, _ in confidences[:n]]

    return images_to_label

def strategy_n_first(image_folder_path: str, imgExtension: str, val_size: int, n: int = DEFAULT_SUB_SAMPLE, **kwargs) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'N first strategy !')
    path_list.sort()
    output_list = path_list[:n]
    return output_list

def strategy_best_entropy(image_folder_path: str,entropy_file :str, imgExtension: str, val_size: int, n: int = DEFAULT_SUB_SAMPLE, **kwargs) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param entropy_file: path to the entropy file
    :return output_list: a list containing the selected images path
    """
    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'N first strategy !')
    path_list.sort()
    data = pd.read_csv(entropy_file, header=None)[0].tolist()  # .values
    idx_list = [i for i in range(len(data))]
    zipped = zip(data, idx_list)
    data_sorted = sorted(zipped, reverse=True)[0:n]
    list_sorted = list(map(list, zip(*data_sorted)))
    idx_frame_sorted = list_sorted[1]
    idx_frame_sorted.sort()
    output_list = []
    for frame_idx in idx_frame_sorted:
        path_construction = f'frame_{frame_idx:04d}'
        output_list.append(path_construction)


    return output_list



def strategy_random(image_folder_path: str, imgExtension: str, val_size: int, n: int = DEFAULT_SUB_SAMPLE, **kwargs) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f'random strategy !')
    path_list.sort()
    output_list = random.sample(path_list, n)
    output_list.sort()
    return output_list


def strategy_fixed_interval(image_folder_path: str, imgExtension: str, val_size: int, n: int = 1, **kwargs) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :return output_list: a list containing the selected images path
    """
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')
    
    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
    
    if n > len(path_list):
        raise SamplingException(f'Image bank contains {len(path_list)} frames, but {n} frames where required for the '
                                f' fixed interval strategy !')
    path_list.sort()   
    step = len(path_list)//n   
    indices = np.arange(0,n)*step
    output_list=[path_list[s] for s in indices]
    return output_list


def strategy_dense_optical_difference(image_folder_path: str, imgExtension: str, val_size: int, n: int = DEFAULT_SUB_SAMPLE, jump = 2, fill_missing: bool = True,
            difference_ratio: float = 4., fixer_precision: float = 0.001, **kwargs) -> list:
    """
    :param image_folder_path: path to the bank image folder
    :param n: number of frames to select
    :param fill_missing: if the number of frames found is less than n, fill with random frames. 
    :param difference_ratio: ratio of changes in the frame
    :param fixer_precision: for fixing outliers (like the entire camera moving from wind). best choices are less or equal to 0.01
    :return output_list: a list containing the selected images path (check for list len if fill_missing is set to False)
    """
    if n <= 0:
        raise SamplingException(f'You must select a strictly positive number of frames to select')

    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
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


def strategy_flow_interval_mix(image_folder_path: str, imgExtension: str, val_size: int, n: int = DEFAULT_SUB_SAMPLE,
            movement_percent: int = 90, difference_ratio: float = 4., **kwargs) -> list:
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

    path_list = list_files_without_extensions(image_folder_path, extension=imgExtension)[0:-val_size]
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

def generate_path_to_frequencies(image_folder_path: str, imgExtension: str):
    
    os.chdir(image_folder_path)
    dic = {}
    count1 = 0
    for file in glob.glob(os.path.join(image_folder_path, "*."+imgExtension)):
        print(str(count1)+  "- Processing image " + str(file))
        sumFrequency_original, sumFrequency_removed = Frequency(image_path=file)
        dic[file.replace(imgExtension,"")] = [np.absolute(sumFrequency_original), np.absolute(sumFrequency_removed)]
        count1=count1+1
    sumFrequencyDataset = pd.DataFrame.from_dict(dic, orient='index', columns=['frequency','frequency_filtered'])
    path_to_store = os.path.join(bank_folder_path, 'frequencies.txt')
    sumFrequencyDataset.to_csv(path_to_store, sep='\t')
    return path_to_store


def diversify_classes(counts, n:int = DEFAULT_SUB_SAMPLE):
    #STEP 1 - Computing the ratios
    ratios = counts/(sum(counts)) # The ratio of the number of instance per class
    dis = np.round(ratios*n) # The distribution without correction
    
    # STEP 2. Sanity Check # 1 - Compensentating rounding errors. The sum in dis may not equal n but will always be inferior. 
    if(sum(dis)<n):
        index_min = np.where(dis==np.min(dis)) #We try to sele
        toBalance = n-sum(dis) #We check the number of missing 
        for c in index_min[0]:
            if(toBalance>0):
                dis[c]=1
                toBalance=toBalance-1
        #Diversify 
    
    index_max = np.where(dis==np.max(dis))
    class_with_zeros = np.where(dis==0.)
    #Balancing null elements
    if((np.max(dis)<n) & (np.any(class_with_zeros))):
        print('Must and can balance frequence class')
        for c in class_with_zeros[0]:
            if(dis[index_max]>1):
                dis[index_max]=dis[index_max]-1
                dis[c]=1
            else:
                print('Imperfect balance')
    
    #Sanity checks
    #assert np.all((dis>=0) | np.all(dis<0)),'Issue with the distribution, it contains negative assignation'
    #assert sum(dis)==n, 'The sum of the distribution does not match the wanted sample'
    return dis

def strategy_frequency(image_folder_path: str, bank_folder_path : str, imgExtension: str, n_groups : int = 10, n: int = DEFAULT_SUB_SAMPLE, **kwargs):
    """
    : param image_folder_path: path to the bank image folder
    : param n: number of frames to select
    : param n_groups : number of wanted clusters.
    : return output_list: a list containing the selected images path
    """
    if(exists(os.path.join(bank_folder_path, 'frequencies.txt'))):
        path_to_frequencies = os.path.join(bank_folder_path, 'frequencies.txt')
    else:
        warning('Must generate the frequencies, the processus takes time')
        path_to_frequencies = generate_path_to_frequencies(image_folder_path, bank_folder_path, imgExtension)
        
    
    df = pd.read_csv(path_to_frequencies, sep='\t',index_col=0)
    sortedDataFiltered = df.sort_values(by=['frequency_filtered'], ascending=False)
    groups = 10 #Hyperparameter of the method. It gives the number of expected cluster. 
    (counts, bins) = np.histogram(sortedDataFiltered.get('frequency_filtered'),groups) #Bins are the edges of the different cluster 
    
    #STEP 2 - Generation of conditions for clustering.
    #Generation of conditions for clustering. : EXAMPLE of the output for 3 groups. 
    '''
    condlist = [(sortedDataFiltered.frequency_filtered  >= bins[0]) & (sortedDataFiltered.frequency_filtered<=bins[1]), 
            (sortedDataFiltered.frequency_filtered  > bins[1]) & (sortedDataFiltered.frequency_filtered<=bins[2]), 
            (sortedDataFiltered.frequency_filtered  > bins[2]) & (sortedDataFiltered.frequency_filtered<=bins[3])]
    '''
    
    condlist = [None] * len(counts) #Initalization of the condition list to assign the group
    for l in range(0,len(bins)-1):
        condlist[l] = (sortedDataFiltered.frequency_filtered  >= bins[l]) & (sortedDataFiltered.frequency_filtered<=bins[l+1]) #@Fixme, In this version the borders's class will be outwritten. We don't care as the method is not sensitive to borders. 
    
    #STEP 3 - Clustering
    condarray = np.array(condlist) # bool representation where each row is a condition and each column is a row of the df
    cond_true = [np.where(i)[0] for i in condarray.T]
    sortedDataFiltered['group']=cond_true    
    
    #Step 4 - Inter-cluster diversification
    dis = diversify_classes(counts,n) #Distribution
    
    #STEP 5 - Selection
    selected_images = pd.DataFrame()
    for i in range(0,len(dis)):
        to_select = sortedDataFiltered[sortedDataFiltered['group']==i].head(int(dis[i]))
        selected_images = selected_images.append(to_select)
    print(selected_images)
    return list(selected_images.index)
