import os

from hydra.utils import call
from .utils import copy_subsample, list_files_without_extensions


def build_val_folder(
    bank_folder, labels_folder, extension, val_set_size=300, min_n_frame=500
):
    validationSet = list_files_without_extensions(
        bank_folder + "/images", extension=extension
    )[-val_set_size::]

    outfolder = os.getcwd()
    if not os.path.exists(f"{outfolder}/val"):
        os.makedirs(f"{outfolder}/val")
    if not os.path.exists(f"{outfolder}/val/images"):
        os.makedirs(f"{outfolder}/val/images")
    if not os.path.exists(f"{outfolder}/val/labels"):
        os.makedirs(f"{outfolder}/val/labels")

    val_folder = outfolder + "/val/"
    copy_subsample(
        validationSet,
        bank_folder,
        val_folder,
        imgExtension=extension,
        labelsFolder=labels_folder,
    )
    return val_folder


def build_train_folder(config):
    subsample_names = call(config.strategy)
    copy_subsample(subsample_names, **config.subsample)
    return config.subsample.out_folder
