import os
import glob
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor



TRANSFORMERS_MODELS = ['rtdetr_v2_r18vd', 'rtdetr_v2_r101vd']

vehicules = [2, 5, 7]

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
        default="rtdetr_v2_r101vd",
        help="Model used to generate the pseudo labels. Default is 'rtdetr_v2_r101vd",
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
    

    if args.model_name not in TRANSFORMERS_MODELS:
        print("NOT RT MODEL, double check the output")
    else:
        print("USING COCO CLASES")
    # images
    img_dir = f"{args.folder}/images"
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.{args.extension}")))

    # labels
    #labels_dir = f"{args.folder}/labels_{args.model_name}"
    #os.makedirs(labels_dir, exist_ok=True)

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

    torch.cuda.set_device(device)

    image_processor = RTDetrImageProcessor.from_pretrained(f"PekingU/{args.model_name}",
                                                           use_fast = True)
    model = RTDetrV2ForObjectDetection.from_pretrained(f"PekingU/{args.model_name}").to(device)

    
    for i in tqdm(range(len(imgs))):
        img_name = os.path.basename(imgs[i]).split(f".{args.extension}")[0]
        try:
            image = Image.open(imgs[i])
            inputs = image_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            W  = image.width
            H  = image.height
            results = image_processor.post_process_object_detection(outputs, 
                                                                    target_sizes=torch.tensor([(H, W)]), 
                                                                    threshold=0.5) # return boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            
            str_data = ""
            for result in results:
                for cls, box, conf in zip(result["labels"], result["boxes"], result["scores"]):       
                    x1, y1, x2, y2 = box
                    xc = ((x1 + x2) / 2) / W
                    yc = ((y1 + y2) / 2) / H
                    w  = (x2 - x1) / W
                    h  = (y2 - y1) / H 
                    box = list((xc, yc, w, h))
                    conf, cls = conf.item(), cls.item() 
                    #box = [round(i, 2) for i in box.tolist()]
                    if cls in vehicules:
                        if not args.output_conf:
                            str_data += f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
                        else:
                            str_data += f"0 {box[0]} {box[1]} {box[2]} {box[3]} {conf}\n"
            
            with open(os.path.join(labels_dir, f"{img_name}.txt"), mode="w") as f:
                f.write(str_data)
        except Exception as e:
            # Log error and image name
            print(f"Error processing image {img_name}: {e}")


if __name__ == "__main__":
    generate_pseudo_labels()
