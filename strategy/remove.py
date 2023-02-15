import os, os.path
import utils
import argparse 
import glob 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--folder_path', type=str, required=True,
                    help='The path to the camera folder, \
                    containing "bank", "train", "val", "test" folders')
    args = ap.parse_args()                
    bank_folder = os.path.join(args.folder_path, 'bank')
    bank_imgs_folder = os.path.join(bank_folder, 'images')
    train_folder = os.path.join(args.folder_path, 'train')
    train_imgs_folder = os.path.join(train_folder, 'images')
    toRemove = os.listdir(train_imgs_folder)
    bank_list = os.listdir(bank_imgs_folder)
    print(len(bank_list))
    print(len(toRemove))
    
    for f in toRemove:
        os.remove(os.path.join(bank_imgs_folder,f))
    
    print(len(bank_list))
    #if(len(files)>0):
        #for f in files:
            #os.remove(f)

    