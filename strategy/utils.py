from logging import warning
import os, os.path
import shutil
import glob 
class SamplingException(Exception):
    pass

def copy_subsample(index,in_folder,out_folder):
    """
    :param index: an array of the name of the images that are selected ('e.g. ['frame_0001','frame_0020'])
    :param in_folder: path to the directory of the source folder containing images and labels subfolders (e.g., "C:/banks")
    :param out_folder: path to the dest (e.g., "C:/train")

    Create a new directory that copies all the images and the labels following the index in a new folder 
    """
    files = glob.glob(f'{out_folder}/*/*')
    if(len(files)>0):
        warning("Train folder was flushed. All files were removed")
        for f in files:
            os.remove(f)
    
    images = os.listdir(os.path.join(in_folder,"images")) #Source of the bank images
    labels = os.listdir(os.path.join(in_folder,"labels")) # Source of the bank of labels
    
    os.makedirs(os.path.join(out_folder,"images"), exist_ok=True) #Create image directory in out_folder if it doesn't exist in out_folder
    os.makedirs(os.path.join(out_folder,"labels"), exist_ok=True) #Create labels directory in out_folder if it doesn't exist in out_folder
    
    for img in index:
        img_with_extension = img+".png"
        img_with_label = img+".txt"
        assert(img_with_extension in images),'Source bank folder does not contain image with name file - ' + img_with_extension
        assert(img_with_label in labels),'Source folder does not contain a file - ' + img_with_label
        shutil.copy(os.path.join(in_folder,"images",img_with_extension),os.path.join(out_folder,"images",img_with_extension))
        shutil.copy(os.path.join(in_folder,"labels",img_with_label),os.path.join(out_folder,"labels",img_with_label))



def create_log_file(folder_path,name,frames_list):
    output_folder = os.path.dirname(folder_path) #EXAMPLE IF FOLDER PATH 'E:/dataset/S05C016', it gives the "E:/dataset"
    video_name = os.path.basename(folder_path)  #EXAMPLE IF FOLDER PATH 'E:/dataset/S05C016', it gives the "S05C016
    with open(os.path.join(output_folder, video_name + '-' + name + ".txt"), "w") as log:
        for f in frames_list:
            log.write(video_name + '/bank/' + f + ".png\n")
