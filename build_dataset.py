#!/usr/bin/env python3
"""
convert2Json.py
================
Convert a YOLO-format dataset (single-class) to COCO JSON **and** optionally build a
Hugging Face *datasets* object from the resulting COCO file.

This script now provides two high‑level helpers:

* **`yolo_to_coco_oneclass(labels_dir, images_dir, category_name)`** – as before.
* **`build_hf_dataset_from_coco(json_path, images_dir)`** – NEW: returns a
  `datasets.Dataset` with columns `image`, `bboxes`, `category_ids` for object
  detection.

A convenience CLI converts the *train* & *val* splits to COCO and can
(optionally) export each split to Arrow after building the HF dataset:

```bash
python convert2Json.py /data/dataset_root \
       --export_hf  # writes HF datasets alongside the JSONs
```

Requirements
------------
* Python 3.8+
* **opencv‑python** *or* Pillow (for image sizes / loading)
* **datasets** >= 2.18.0

Install extra deps:
```bash
pip install pillow datasets
```
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
    from PIL import Image  # type: ignore


try:
    from datasets import Dataset, Features, Sequence, Value, DatasetDict, Image as HFImage  # type: ignore
except ImportError:
    Dataset = None  # type: ignore

__all__ = [
    "yolo_to_coco_oneclass",
    "build_hf_dataset_from_coco",
    "convert_dataset",
]

# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _image_size(path: str) -> Tuple[int, int]:
    """Return *(width, height)* for *path* using cv2 or Pillow."""
    if cv2 is not None:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Unable to read image: {path}")
        h, w = img.shape[:2]
        return w, h
    with Image.open(path) as im:
        w, h = im.size
    return w, h


# ---------------------------------------------------------
# 1. YOLO → COCO (single‑class)
# ---------------------------------------------------------

def yolo_to_coco_oneclass(
    labels_dir: str,
    images_dir: str,
    category_name: str = "object",
) -> Tuple[str, Dict]:
    """Convert YOLO annotations in *labels_dir* to COCO JSON for a single class.

    Returns *(json_path, coco_dict)*.
    """

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories = [{"id": 0, "name": category_name}]

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted(
        f for f in glob.glob(os.path.join(images_dir, "*")) if Path(f).suffix.lower() in supported_ext
    )

    ann_id = 0
    for img_id, img_path in enumerate(img_files):
        file_name = Path(img_path).name
        width, height = _image_size(img_path)

        images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        label_path = os.path.join(labels_dir, f"{Path(file_name).stem}.txt")
        if not os.path.isfile(label_path):
            continue  # image without annotations

        with open(label_path, "r", encoding="utf-8") as fp:
            for line in fp:
                if not (line := line.strip()):
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Malformed line in {label_path}: '{line}' (expected 5 elements)"
                    )
                cls, cx, cy, bw, bh = map(float, parts)
                if int(cls) != 0:
                    raise ValueError(
                        f"Non‑zero class id {cls} in single‑class dataset ({label_path})"
                    )
                x_min = (cx - bw / 2) * width
                y_min = (cy - bh / 2) * height
                box_w = bw * width
                box_h = bh * height

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": [x_min, y_min, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    coco_dict: Dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    json_path = os.path.abspath(os.path.join(labels_dir, "labels_coco.json"))
    os.makedirs(labels_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(coco_dict, fp, indent=2)

    return json_path, coco_dict


# ---------------------------------------------------------
# 2. COCO → Hugging Face Dataset
# ---------------------------------------------------------

def build_hf_dataset_from_coco(
    json_path: str,
    images_dir: str,
    *,
    include_empty: bool = True,
) -> "Dataset":
    """Build a **Hugging Face datasets** object from a COCO JSON + images dir.

    The resulting dataset has these columns:
        * **id**: `int64` – image id (COCO).
        * **image**: `datasets.Image` – auto‑decoded when accessed.
        * **bboxes**: `(n, 4) float32` – [x_min, y_min, width, height].
        * **category_ids**: `sequence<int64>` – parallel list of class ids.

    Args
    ----
    json_path: Path to COCO JSON file.
    images_dir: Directory containing the image files referenced in *json_path*.
    include_empty: Keep images without annotations? (default **True**)

    Returns
    -------
    datasets.Dataset
    """

    if Dataset is None:
        raise ImportError(
            "`datasets` library not installed – `pip install datasets` to use build_hf_dataset_from_coco()"
        )

    with open(json_path, "r", encoding="utf-8") as fp:
        coco = json.load(fp)

    # Map image_id → image metadata + empty anno lists
    records: Dict[int, Dict] = {
        img["id"]: {
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "bboxes": [],
            "category_ids": [],
            "areas": [],
            "iscrowds": [],
        }
        for img in coco["images"]
    }

    for ann in coco["annotations"]:
        rec = records[ann["image_id"]]
        rec["bboxes"].append(ann["bbox"])
        rec["category_ids"].append(ann["category_id"])
        rec["areas"].append(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
        rec["iscrowds"].append(ann.get("iscrowd", 0))

    if not include_empty:
        records = {k: v for k, v in records.items() if v["bboxes"]}

    # Hugging Face expects absolute paths for Image feature
    for rec in records.values():
        abs_path = os.path.join(images_dir, rec["file_name"])
        rec["file_path"] = abs_path
        rec["image"] = abs_path

        # pack HF-object‑detection style structure
        rec["objects"] = {
            "bbox": rec["bboxes"],
            "category_id": rec["category_ids"],
            "area": rec["areas"],
            "iscrowd": rec["iscrowds"],
        }
        # remove the flat lists (keep only nested)
        rec.pop("bboxes")
        rec.pop("category_ids")
        rec.pop("areas")
        rec.pop("iscrowds")
        # drop unused keys
        rec.pop("file_name")
        rec.pop("width")
        rec.pop("height")

    # build dataset
    features = Features(
        {
            "id": Value("int64"),
            "image": HFImage(),
            "file_path": Value("string"),
            "objects": {
                "bbox": Sequence(
                    feature=Sequence(Value("float32"), length=4)
                ),
                "category_id": Sequence(Value("int64")),
                "area": Sequence(Value("float32")),
                "iscrowd": Sequence(Value("int64")),
            },
        }
    )

    ds = Dataset.from_list(list(records.values()), features=features)
    return ds


# ---------------------------------------------------------
# 3. Convenience batch converter (YOLO → COCO → HF)
# ---------------------------------------------------------

def convert_dataset(path_run: str) -> DatasetDict:

    train_path_json_labels, train_json = yolo_to_coco_oneclass(
        os.path.join(path_run, "train", "labels"),
        os.path.join(path_run, "train", "images")
    )
    val_path_json_labels, val_json = yolo_to_coco_oneclass(
        os.path.join(path_run, "val", "labels"),
        os.path.join(path_run, "val", "images")
    )


    train_ds = build_hf_dataset_from_coco(train_path_json_labels, os.path.join(path_run, "train", "images"))

    val_ds = build_hf_dataset_from_coco(val_path_json_labels, os.path.join(path_run, "val", "images"))
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    return dataset



