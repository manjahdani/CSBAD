import os, os.path
import glob
import shutil
from logging import warning
import copy

class SamplingException(Exception):
    pass



def find_file_extension(directory):
    """
    Finds the file extension of the first image file in the given directory, excluding the dot.
    Assumes all image files in the directory have the same extension.
    """
    for file in os.listdir(directory):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            return os.path.splitext(file)[1].lstrip('.')  # Removes the dot from the extension
    raise RuntimeError(f"No suitable image file found in {directory} for extension determination")




def list_files_without_extensions(path: str) -> list:
    """
    :param path: path to scan for files
    :param extensions: what type of files to scan for
    :return path_list: list of file names without the extensions
    """

    extension = find_file_extension(path)

    path_list = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(path)
        if filename.endswith(extension)
    ]
    return path_list, extension









