import torch
import glob
from _utils import *
import re
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, required=True, help='Path to the parent dir')
parser.add_argument('--r', type=int, required=False, default=1920, help='Model input width resolution. Default is 1920')
parser.add_argument('--e', type=str, required=False, default='png', help='Image extension. Default is png')
parser.add_argument('--t', type=bool, required=False, default=True, help='Amalgamize trucks as cars. Default is True')
args = parser.parse_args()

parent_dir = args.p

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', _verbose=True)  # , 'custom', path='path/to/best.pt'
# model.multi_label = True
model.classes = [2, 3, 5, 7]  # keep car,motorcycle,bus,truck
model.conf = 0.4
model.iou = 0.7
model.agnostic = True

# Images

img_dir = f'{parent_dir}/images'
lbl_dir = f'{parent_dir}/labels'

imgs = sorted(glob.glob(f'{img_dir}/*.{args.e}'))
imgs = [img.replace('\\', '/') for img in imgs]
# Inference
for i in tqdm(range(len(imgs))):
    img_name = re.search(f'{img_dir}/(.+?).{args.e}', imgs[i]).group(1)
    results = model(imgs[i], size=args.r)  # imgs[i],size=1200
    to_show_rgb = results.render()[0]
    H, W, C = to_show_rgb.shape
    to_save = df_to_txt(results.pandas().xyxy[0], H, W, amalgamize_truck=args.t)
    np.savetxt(f'{lbl_dir}/{img_name}.txt', to_save.values, fmt='%d %f %f %f %f')  #
