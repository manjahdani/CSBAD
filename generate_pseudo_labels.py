#s
import os
import sys
import glob
import argparse
from tqdm import tqdm
import torch

sys.path.append(os.path.join(sys.path[0], "yolov8", "ultralytics"))
from ultralytics import YOLO

COCO_MODELS=['yolov8n','yolov8x6','yolov8s','yolov8l','yolov8m']
def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, required=True, help="Path to the folder directory"
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="jpg",
        help="Image extension. Default is 'jpg'",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8x6",
        help="Model used to generate the pseudo labels. Default is 'yolov8x6'",
    )
    parser.add_argument(
        "--output-conf",
        action="store_true",
        default=False,
        help="Output confidences. Default is False",
    )
    return parser.parse_args()


def generate_pseudo_labels():
    args = handle_args()

    """ Vehicles in COCO dataset (80 classes)
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
    model = YOLO(f"{args.model_name}.pt", type="v8")

    if args.model_name not in COCO_MODELS:
        print("NOT COCO MODEL, double check the output")
    else:
        print("USING COCO CLASES")
    # images
    img_dir = f"{args.folder}/images"
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.{args.extension}")))

    # labels
    labels_dir = f"{args.folder}/labels_{args.model_name}"
    os.makedirs(labels_dir, exist_ok=True)

    if not args.output_conf:
        labels_dir = f"{args.folder}/labels_{args.model_name}"
        os.makedirs(labels_dir, exist_ok=True)
    else:
        labels_dir = f"{args.folder}/labels_{args.model_name}_w_conf"
        os.makedirs(labels_dir, exist_ok=True)

    # inference
    # Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda:0"  # Use GPU
    else:
        device = None  # Use CPU

    for i in tqdm(range(len(imgs))):
        img_name = os.path.basename(imgs[i]).split(f".{args.extension}")[0]
        results = model.predict(source=imgs[i], verbose=False, device=device)
        boxes = results[0].boxes.xywhn
        classes = results[0].boxes.cls
        confs = results[0].boxes.conf
        str = ""
        for cls, box, conf in zip(classes, boxes, confs):
            if(args.model_name in COCO_MODELS):
                if cls in vehicules:
                    if not args.output_conf:
                        str += f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
                    else:
                        str += f"0 {box[0]} {box[1]} {box[2]} {box[3]} {conf}\n"
            else:
                if not args.output_conf:
                    str += f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
                else:
                    str += f"0 {box[0]} {box[1]} {box[2]} {box[3]} {conf}\n"

        with open(os.path.join(labels_dir, f"{img_name}.txt"), mode="w") as f:
            f.write(str)


if __name__ == "__main__":
    generate_pseudo_labels()
