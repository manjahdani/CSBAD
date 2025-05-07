import os
import yaml
import hydra
import wandb
import torch
import shutil
import random
import numpy as np
from collections.abc import Mapping
from typing import Any, Optional, Union
from PIL import Image
from transformers.image_processing_utils import BatchFeature

from hydra.utils import to_absolute_path
import uuid
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from evalMAP import MAPEvaluator
from transformers import RTDetrV2ForObjectDetection, TrainingArguments, Trainer
from transformers import AutoImageProcessor as RTDetrV2ImageProcessor
from build_dataset import convert_dataset
from subsampling.dataset_builder import build_val_folder, build_train_folder
import albumentations as A

# Version checks
check_min_version("4.52.0.dev0")
require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/object-detection/requirements.txt")

# Preprocessor arguments
IMAGE_SQUARE_SIZE = 640
DO_RESIZE = True
DO_PAD = True
USE_FAST = True

def clip_bbox_coco(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]

@hydra.main(version_base=None, config_path="experiments", config_name="experiment")
def train(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.cam_week_pairs, n_cameras = generate_cameras_pairs(config.dataset.name)

    if config.training_mode == "cst_maturity":
        if config.N_streams != "null":
            config.model.epochs = int(config.model.epochs * config.N_streams / n_cameras)
            config.epochs = config.model.epochs
        else:
            raise ValueError("For constant maturity study, 'N_streams' is required.")

    AUGMENTATION_TRAIN = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),
        A.Resize(IMAGE_SQUARE_SIZE, IMAGE_SQUARE_SIZE),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_id']))

    torch.cuda.set_device(device)
    set_random(config.seed)

    val_folder = build_val_folder(**config.val)
    train_folder = build_train_folder(config.train)
    path_run = update_config_file(config)

    dataset = convert_dataset(path_run)

    checkpoint = f"PekingU/{config.student}"
    image_processor = RTDetrV2ImageProcessor.from_pretrained(
        checkpoint,
        do_resize=DO_RESIZE,
        size={"height": IMAGE_SQUARE_SIZE, "width": IMAGE_SQUARE_SIZE},
        do_pad=DO_PAD,
        pad_size={"height": IMAGE_SQUARE_SIZE, "width": IMAGE_SQUARE_SIZE},
        do_normalize=True,
        do_rescale=True
    )

    def collate_fn(batch: list[BatchFeature]) -> Mapping[str, Union[torch.Tensor, list[Any]]]:
        batch = [x for x in batch if len(x["labels"]) > 0]
        if len(batch) == 0:
            return {}
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        data = {"pixel_values": pixel_values, "labels": [x["labels"] for x in batch]}
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data

    def preprocess_batch(examples):
        images, annotations = [], []

        for i, path in enumerate(examples["file_path"]):
            img = np.array(Image.open(path).convert("RGB"))
            obj = examples["objects"][i]

            if len(obj["bbox"]) == 0:
                continue

            bboxes = obj["bbox"]
            category_ids = obj["category_id"]

            try:
                transformed = AUGMENTATION_TRAIN(image=img, bboxes=bboxes, category_id=category_ids)
            except ValueError:
                continue

            img_h, img_w = transformed["image"].shape[:2]
            ann_list = []
            for bbox, cid in zip(transformed["bboxes"], transformed["category_id"]):
                clipped_bbox = clip_bbox_coco(bbox, img_w, img_h)
                ann_list.append({
                    "category_id": cid,
                    "bbox": clipped_bbox,
                    "area": clipped_bbox[2] * clipped_bbox[3],
                    "iscrowd": 0
                })

            images.append(Image.fromarray(transformed["image"]))
            annotations.append({
                "image_id": examples["id"][i],
                "annotations": ann_list
            })

        if len(images) == 0:
            return {}

        processed = image_processor(images=images, annotations=annotations, return_tensors="pt")
        processed["labels"] = processed.pop("labels")
        return processed

    def preprocess_batch_val(examples):
        images = [Image.open(p).convert("RGB") for p in examples["file_path"]]
        annotations = []

        for i in range(len(images)):
            objs = examples["objects"][i]
            anns = [
                {
                    "category_id": objs["category_id"][j],
                    "bbox": objs["bbox"][j],
                    "area": objs["area"][j],
                    "iscrowd": objs["iscrowd"][j]
                }
                for j in range(len(objs["bbox"]))
            ]
            annotations.append({"image_id": examples["id"][i], "annotations": anns})

        processed = image_processor(images=images, annotations=annotations, return_tensors="pt")
        processed["labels"] = processed.pop("labels")
        return processed

    dataset["train"] = dataset["train"].with_transform(preprocess_batch)
    dataset["validation"] = dataset["validation"].with_transform(preprocess_batch_val)

    model = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=1,
        id2label={0: "vehicle"},
        label2id={"vehicle": 0},
        ignore_mismatched_sizes=True
    ).to(device)

    config.model.name = f"{str(uuid.uuid4())[:8]}_{config.model.name}_{config.model.epochs}"
    train_args = TrainingArguments(
        run_name=config.model.name,
        output_dir=to_absolute_path(os.path.join("models", config.model.name)),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=config.model.epochs,
        max_grad_norm=0.1,
        warmup_steps=300,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_map_50_95",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        greater_is_better=True,
        learning_rate=5e-5,
        weight_decay=1e-4,
        dataloader_num_workers=2,
        fp16=True,
        logging_steps=16,
        load_best_model_at_end=True,
        save_total_limit=1,
        eval_on_start=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_do_concat_batches=False,
        report_to="wandb"
    )

    eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label={0: "vehicle"})

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        data_collator=collate_fn,
        eval_dataset=dataset["validation"],
        compute_metrics=eval_compute_metrics_fn
    )
    image_processor.save_pretrained(to_absolute_path(os.path.join("models", config.model.name)))

    trainer.train()
    wandb.finish()

    #shutil.rmtree(val_folder, ignore_errors=True)
    #shutil.rmtree(train_folder, ignore_errors=True)

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def update_config_file(config):
    with open(config.model.data, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["path"] = os.getcwd()
    with open("data.yaml", mode="w") as f:
        yaml.dump(data, f)
    return data["path"]

def generate_cameras_pairs(input_string):
    dash_index = input_string.index('-')
    week_index = input_string.index('-week')
    cameras = input_string[dash_index + 4:week_index].split('o')
    weeks = input_string[week_index + 5:].split('o')
    if len(cameras) != len(weeks):
        raise ValueError("The number of cameras must be equal to the number of weeks")
    return [{'cam': int(cam), 'week': int(week)} for cam, week in zip(cameras, weeks)], len(cameras)

if __name__ == "__main__":
    train()
