import os

from hydra.utils import call
from .utils import list_files_without_extensions
from logging import warning
import shutil
import glob
class SamplingException(Exception):
    pass


def build_val_folder(cam_week_pairs, base_folder, labels_folder, extension, val_set_size=300, teacher="yolov8x6"):
    assert len(cam_week_pairs) > 0, "At least one camera-week pair should be specified"
    val_samples_per_cam = val_set_size// len(cam_week_pairs)
    
    outfolder = os.getcwd()
    if not os.path.exists(f"{outfolder}/val"):
        os.makedirs(f"{outfolder}/val")
    if not os.path.exists(f"{outfolder}/val/images"):
        os.makedirs(f"{outfolder}/val/images")
    if not os.path.exists(f"{outfolder}/val/labels"):
        os.makedirs(f"{outfolder}/val/labels")

    val_folder = outfolder + "/val/"
    
    files = glob.glob(f"{val_folder}/*/*")
    if len(files) > 0:
        print('to be flushed',val_folder)
        warning("Train folder was flushed. All files were removed")
        for f in files:
            os.remove(f)
    for pair in cam_week_pairs:
        cam, week = pair['cam'], pair['week']
        print(f"Starting copy validation set for cam ${cam} and week ${week}")
        bank_folder = os.path.join(base_folder, f"cam{cam}", f"week{week}", "bank")
        labels_folder = os.path.join(bank_folder, f"labels_{teacher}")
        validation_set = list_files_without_extensions(
            bank_folder + "/images", extension=extension
        )[-val_samples_per_cam:]
        copy_subsample(
            validation_set,
            bank_folder,
            val_folder,
            imgExtension=extension,
            labelsFolder=labels_folder,
        )
    return val_folder


def build_train_folder(config):
    assert len(config.cam_week_pairs) > 0, "At least one camera-week pair should be specified"

    train_samples_per_cam = config.strategy.n // len(config.cam_week_pairs)
    config.strategy.n=train_samples_per_cam 
    
    files = glob.glob("train/*/*")
    if len(files) > 0:
        print('to be flushed',"train")
        warning("Train folder was flushed. All files were removed")
        for f in files:
            os.remove(f)
    
    for pair in config.cam_week_pairs:
        cam, week = pair['cam'], pair['week']
        bank_folder = os.path.join(config.base_folder, f"cam{cam}", f"week{week}", "bank")
        labels_folder = os.path.join(bank_folder, f"labels_{config.teacher}")
        image_folder = bank_folder + "/images"
        config.strategy.image_folder_path=image_folder
        config.strategy.image_labels_path=os.path.join(bank_folder,f"labels_{config.student}_w_conf")
        subsample_names = call(config.strategy)
        copy_subsample(
            subsample_names,
            bank_folder,
            "train",
            imgExtension=config.extension,
            labelsFolder=labels_folder,
        )
    return "train"


def copy_subsample(index, in_folder, out_folder, imgExtension, labelsFolder):
    """
    :param index: an array of the name of the images that are selected ('e.g. ['frame_0001','frame_0020'])
    :param in_folder: path to the directory of the source folder containing images and labels subfolders (e.g., "C:/banks")
    :param out_folder: path to the dest (e.g., "C:/train")

    Create a new directory that copies all the images and the labels following the index in a new folder
    """
    images = os.listdir(os.path.join(in_folder, "images"))  # Source of the bank images
    labels = os.listdir(
        os.path.join(in_folder, labelsFolder)
    )  # Source of the bank of labels
    print(f"Copying {len(index)} images from",os.path.join(in_folder, "images"), f"containing {len(images)} images and labels {len(labels)}")
    os.makedirs(
        os.path.join(out_folder, "images"), exist_ok=True
    )  # Create image directory in out_folder if it doesn't exist in out_folder
    os.makedirs(
        os.path.join(out_folder, "labels"), exist_ok=True
    )  # Create labels directory in out_folder if it doesn't exist in out_folder
    
    for img in index:
        img_with_extension = img + str(".") + imgExtension
        img_with_label = img + ".txt"
        assert img_with_extension in images, (
            "Source bank folder does not contain image with name file - "
            + img_with_extension
        )
        assert img_with_label in labels, (
            "Source folder does not contain a file - " + img_with_label
        )
        shutil.copy(
            os.path.join(in_folder, "images", img_with_extension),
            os.path.join(out_folder, "images", img_with_extension),
        )
        shutil.copy(
            os.path.join(in_folder, labelsFolder, img_with_label),
            os.path.join(out_folder, "labels", img_with_label),
        )

