import os
import shutil

from glob import glob
from hydra.utils import call


def copy_subsample(subsample_names, images_folder, labels_folder, out_folder, imgExtension):
    images = os.listdir(images_folder) #Source of the bank images
    labels = os.listdir(labels_folder) # Source of the bank of labels
    
    os.makedirs(os.path.join(out_folder,"images"), exist_ok=True) #Create image directory in out_folder if it doesn't exist in out_folder
    os.makedirs(os.path.join(out_folder,"labels"), exist_ok=True) #Create labels directory in out_folder if it doesn't exist in out_folder
    
    for name in subsample_names:
        img_with_extension = name + f".{imgExtension}"
        label_with_extension = name + ".txt"
        assert(img_with_extension in images),'Source bank folder does not contain image with name file - ' + img_with_extension
        assert(label_with_extension in labels),'Source folder does not contain a file - ' + label_with_extension
        shutil.copy(os.path.join(images_folder, img_with_extension),
                    os.path.join(out_folder, "images", img_with_extension))
        shutil.copy(os.path.join(labels_folder, label_with_extension),
                    os.path.join(out_folder, "labels", label_with_extension))
    print(f"{len(subsample_names)} images + labels have been copied from source to destination.")


def build_train_folder(images_folder, labels_folder, img_extension, cfg_strategy):
    subsample_names = call(cfg_strategy)

    out_folder = os.path.join(os.getcwd(), "train")
    os.makedirs(out_folder, exist_ok=True)
    copy_subsample(subsample_names, images_folder, labels_folder, out_folder, img_extension)
    return out_folder


def build_val_folder(images_folder, labels_folder, img_extension, val_set_size):
    list_images = []
    list_images.extend(glob(os.path.join(images_folder, f"*.{img_extension}")))
    assert len(list_images) > 0, "No images found in the validation folder"
    assert len(list_images) >= val_set_size, "Not enough images found in the validation folder"
    val_list_images = list_images[-val_set_size:]
    assert len(val_list_images) == val_set_size, "Problem with the validation set size"
    subsample_names = [os.path.splitext(os.path.basename(image))[0] for image in val_list_images]

    out_folder = os.path.join(os.getcwd(), "val")
    os.makedirs(out_folder, exist_ok=True)
    copy_subsample(subsample_names, images_folder, labels_folder, out_folder, img_extension)
    return out_folder