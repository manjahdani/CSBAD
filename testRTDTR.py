#!/usr/bin/env python3
"""evaluate_rt_detr_torchmetrics.py

Batch‑evaluate RT‑DETR‑v2 checkpoints with **torchmetrics.MeanAveragePrecision**
-------------------------------------------------------------------------------
* Drops the dependency on *pycocotools* (still optional).
* Works out‑of‑the‑box for single‑class RT‑DETR fine‑tuned models.
* Produces a CSV with one line per (checkpoint × camera) and stores
  the global mAP@[.5:.95] value computed by torchmetrics.

Usage
-----
$ python evaluate_rt_detr_torchmetrics.py

Environment
-----------
* `torchmetrics>=1.3`
* `torch         >=2.0`
* `transformers  >=4.37` (RT‑DETR‑v2 support)

"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from transformers import (AutoImageProcessor,
                          RTDetrV2ForObjectDetection)

# ------------------------- paths & constants -------------------------
MODELS_ROOT_DIR = Path("./models")
SAVE_VIS_DIR    = Path("./visualizations_eval")
PRED_SAVE_DIR   = Path("./predictionsForEval")
RESULT_CSV      = Path("eval_rt_results.csv")

SAVE_VIS_DIR.mkdir(exist_ok=True)
PRED_SAVE_DIR.mkdir(exist_ok=True)

CSV_HEADER = [
    "run_id", "Source_dataset", "Target_domain",
    "Student_Type", "Version_Student", "Student_Arch",
    "Teacher_Type", "Version_Teacher", "Teacher_Arch",
    "Strategy", "Active‑Learning‑Setting",
    "Total_Samples", "epochs", "mAP50‑95"
]

# ------------------------- helpers -------------------------

def parse_run_folder_name(folder_name: str) -> List[str] | None:
    parts = folder_name.split("_")
    return parts if len(parts) == 12 else None


def coco_to_torchmetrics_targets(coco_json: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load a COCO GT JSON and return a dict: image_id → {boxes, labels}."""
    with open(coco_json, "r", encoding="utf-8") as fp:
        coco = json.load(fp)

    targets: Dict[int, Dict[str, List]] = {}
    for img in coco["images"]:
        targets[img["id"]] = {"boxes": [], "labels": []}
    for ann in coco["annotations"]:
        tgt = targets[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        tgt["boxes"].append([x, y, x + w, y + h])
        tgt["labels"].append(ann["category_id"])

    # convert to tensors
    for k, v in targets.items():
        v["boxes"]  = torch.tensor(v["boxes"], dtype=torch.float32)
        v["labels"] = torch.tensor(v["labels"], dtype=torch.int64)
    return targets


# ------------------------- main loop -------------------------

def main() -> None:
    # existing results cache
    existing_rows = set()
    if RESULT_CSV.exists():
        with open(RESULT_CSV, newline="") as f:
            reader = csv.reader(f)
            if next(reader, None) != CSV_HEADER:
                RESULT_CSV.write_text("")
            else:
                for row in reader:
                    existing_rows.add(tuple(row[:13]))
    else:
        with open(RESULT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    for cam_idx in range(1, 10):
        target_domain = f"cam{cam_idx}"
        test_dir      = Path(f"/home/dani/data/data/WALT-challenge/{target_domain}/test")
        img_dir       = test_dir / "images"
        gt_json       = test_dir / "dataset.json"

        print(f"\n=== {target_domain}: {gt_json}")
        coco_gt = json.load(open(gt_json))
        image_meta = {img["file_name"]: {"id": img["id"], "width": img["width"], "height": img["height"]} for img in coco_gt["images"]}
        id2anns = {}
        for ann in coco_gt["annotations"]:
            id2anns.setdefault(ann["image_id"], []).append(ann)

        for run_folder in sorted(MODELS_ROOT_DIR.iterdir()):
            if not run_folder.is_dir():
                continue
            meta = parse_run_folder_name(run_folder.name)
            if meta is None:
                print(f"[skip] {run_folder.name}: bad name")
                continue
            row_no_metric = meta[:1] + [meta[1], target_domain] + meta[2:]
            row_key = tuple(row_no_metric)
            if row_key in existing_rows:
                print(f"[skip] done: {run_folder.name} on {target_domain}")
                continue

            # locate HF checkpoint sub‑dir
            ckpt = next((d for d in run_folder.iterdir() if d.is_dir() and "checkpoint" in d.name), None)
            if ckpt is None:
                print(f"[skip] {run_folder.name}: no checkpoint dir")
                continue

            print(f"→ evaluating {run_folder.name} on {target_domain}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoImageProcessor.from_pretrained(ckpt)
            model     = RTDetrV2ForObjectDetection.from_pretrained(ckpt).to(device).eval()

            metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

            preds_json: List[dict] = []
            vis_written = 0

            for fname in tqdm(sorted(img_dir.iterdir()), desc=f"{run_folder.name}/{target_domain}"):
                if fname.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                image = Image.open(fname).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model(**inputs)

                processed = processor.post_process_object_detection(
                    output, threshold=0.5,
                    target_sizes=torch.tensor([[image_meta[fname.name]["height"], image_meta[fname.name]["width"]]], device=device)
                )[0]

                if fname.name not in image_meta:
                    continue
                meta = image_meta[fname.name]
                image_id = meta["id"]
                if image_id is None:
                    continue

                # prepare tensors for torchmetrics
                preds_dict = {
                    "boxes":  processed["boxes"].cpu(),
                    "scores": processed["scores"].cpu(),
                    "labels": processed["labels"].cpu(),
                }
                target_boxes = [ann["bbox"] for ann in id2anns.get(image_id, [])]
                target_boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in target_boxes], dtype=torch.float32)
                target_labels = torch.tensor([ann["category_id"] for ann in id2anns.get(image_id, [])], dtype=torch.int64)
                targets_dict = {"boxes": target_boxes, "labels": target_labels}
                metric.update([preds_dict], [targets_dict])

                # also save JSON in COCO format for optional later use
                for b, s in zip(processed["boxes"], processed["scores"]):
                    x0, y0, x1, y1 = b.tolist()
                    preds_json.append({
                        "image_id": int(image_id),
                        "category_id": 0,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(s)
                    })

                # write one visualisation per run
                if vis_written < 1:
                    vis = image.copy()
                    draw = ImageDraw.Draw(vis)
                    for ann in id2anns.get(image_id, []):
                        xb, yb, w, h = ann["bbox"]
                        draw.rectangle([xb, yb, xb + w, yb + h], outline="blue", width=2)
                    for b in processed["boxes"]:
                        x0, y0, x1, y1 = b.tolist()
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                    vis.save(SAVE_VIS_DIR / f"{run_folder.name}_{target_domain}_{fname.name}")
                    vis_written += 1

            # ─── final metric ───
            scores = metric.compute()
            map_50_95 = round(float(scores["map"]), 4)
            print(f"mAP@[.5:.95] = {map_50_95}")

            # write CSV
            with open(RESULT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row_no_metric + [map_50_95])
            existing_rows.add(row_key)

            # save preds JSON
            pred_json_path = PRED_SAVE_DIR / f"{run_folder.name}_{target_domain}_pred.json"
            with open(pred_json_path, "w") as fp:
                json.dump(preds_json, fp)


if __name__ == "__main__":
    main()
