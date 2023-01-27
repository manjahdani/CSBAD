import os
import sys
import glob
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "yolov8", "ultralytics"))
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--parent", type=str, required=True, help="Path to the parent dir")
parser.add_argument(
    "--extension",
    type=str,
    required=True,
    help="Image extension. Default is png",
)
args = parser.parse_args()

# vehicles
"""
1: bicycle
2: car
3: motorcycle
4: airplane
5: bus
6: train
7: truck
8: boat
"""
vehicules = [2, 5, 7]

# model
model_name = "yolov8x6"
model = YOLO(f"{model_name}.pt", type="v8")

# images
img_dir = f"{args.parent}/images"
imgs = sorted(glob.glob(os.path.join(img_dir, f"*.{args.extension}")))

# labels
lbl_dir = f"{args.parent}/labels"
os.makedirs(lbl_dir, exist_ok=True)

# inference
for i in tqdm(range(len(imgs))):
    img_name = os.path.basename(imgs[i]).split(f".{args.extension}")[0]
    results = model.predict(source=imgs[i], verbose=False)
    boxes = results[0].boxes.xywhn
    classes = results[0].boxes.cls
    str = ""
    for cls, box in zip(classes, boxes):
        if cls in vehicules:
            str += f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
    with open(os.path.join(lbl_dir, f"{img_name}.txt"), mode="w") as f:
        f.write(str)
