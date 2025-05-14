import csv
import os
import warnings
import datetime
from pathlib import Path
from typing import List, Union

import torch
import pytorch_lightning as pl
from nanodet.util import load_config, cfg, mkdir, NanoDetLightningLogger, convert_old_model
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask

# ------------------------- paths & constants -------------------------
MODELS_ROOT_DIR = Path("./workspace/")
RESULT_CSV      = Path("eval_nanodet_results.csv")
SAVEDIR =  Path("./eval")
CSV_HEADER = [
    "run_id", "Source_dataset", "Source_domain","Source_period", 
    "Target_domain",
    "Student",
    "Teacher",
    "Strategy", "Active‑Learning‑Setting",
    "Samples", "epochs", "mAP50‑95"
]

# ------------------------- helpers -------------------------
def parse_run_folder_name(folder_name: str, target_domain: str) -> Union[List[str], None]:
    parts = folder_name.split("_")
    if len(parts) < 8 or "-" not in parts[1]:
        return None
    source_parts = parts[1].split("-")
    source_dataset = source_parts[0]
    source_domain = source_parts[1]
    source_period = "-".join(source_parts[2:])
    meta = [
        parts[0],               # run_id
        source_dataset,         # Source_dataset
        source_domain,          # Source_domain
        source_period,          # Source_period
        target_domain,          # Target_domain
        parts[2],               # Student
        parts[3],               # Teacher
        parts[4],               # Strategy
        parts[5],               # Active-Learning-Setting
        parts[6],               # Samples
        parts[7]                # epochs
    ]
    return meta


def read_eval_results(cfg):
    result_file = os.path.join(cfg.save_dir, "eval_results.txt")
    results = {}
    if not os.path.exists(result_file):
        print(f"[error] eval_results.txt not found at {result_file}")
        return results

    with open(result_file, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":")
                results[key.strip()] = float(value.strip())

    return results

# ------------------------- main evaluation loop -------------------------
def main():
    existing_rows = set()
    if RESULT_CSV.exists():
        with open(RESULT_CSV, newline="") as f:
            reader = csv.reader(f)
            if next(reader, None) != CSV_HEADER:
                RESULT_CSV.write_text("")
            else:
                for row in reader:
                    existing_rows.add(tuple(row[:-1]))
    else:
        with open(RESULT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    for cam_idx in range(1, 10):
        target_domain = f"cam{cam_idx}"
        test_dir = Path(f"/home/dani/data/WALT-challenge/{target_domain}/test")
        ann_file = str(test_dir / "labels_coco.json")
        img_prefix = str(test_dir / "images")

        for run_folder in sorted(MODELS_ROOT_DIR.iterdir()):
            if not run_folder.is_dir():
                continue
            meta = parse_run_folder_name(run_folder.name, target_domain)
            row_key = tuple(meta)
            if row_key in existing_rows:
                print(f"[skip] done: {run_folder.name} on {target_domain}")
                continue

            config_path = run_folder / "config.yml"
            if not config_path.exists():
                print(f"[skip] {run_folder.name}: config.yml not found at root")
                continue

            ckpt_path = run_folder / "model_best" / "model_best.ckpt"
            if not ckpt_path.exists():
                print(f"[skip] {run_folder.name}: missing checkpoint at model_best/model_best.ckpt")
                continue

            print(f"→ Evaluating {run_folder.name} on {target_domain}")

            # Load and override config
            load_config(cfg, str(config_path))
            cfg.defrost()
            cfg.data.val.update({"ann_path": ann_file, "img_path": img_prefix})
            cfg.test_mode = "val"

            timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            cfg.save_dir = os.path.join(SAVEDIR, f"{run_folder.name}_{target_domain}_{timestr}")
            mkdir(-1, Path(cfg.save_dir))

            logger = NanoDetLightningLogger(cfg.save_dir)

            val_dataset = build_dataset(cfg.data.val, "val")
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.device.batchsize_per_gpu,
                shuffle=False,
                num_workers=cfg.device.workers_per_gpu,
                pin_memory=True,
                collate_fn=naive_collate,
                drop_last=False,
            )
            evaluator = build_evaluator(cfg.evaluator, val_dataset)

            task = TrainingTask(cfg, evaluator)
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            if "pytorch-lightning_version" not in ckpt:
                warnings.warn("Old .pth checkpoint format detected. Converting.")
                ckpt = convert_old_model(ckpt)
            task.load_state_dict(ckpt["state_dict"])

            accelerator = "gpu" if cfg.device.gpu_ids != -1 else "cpu"
            devices = cfg.device.gpu_ids if accelerator == "gpu" else None

            trainer = pl.Trainer(
                default_root_dir=cfg.save_dir,
                accelerator=accelerator,
                devices=devices,
                logger=logger,
                log_every_n_steps=cfg.log.interval,
                num_sanity_val_steps=0,
            )

            logger.info("Starting testing...")
            _ = trainer.test(task, val_dataloader)

            eval_results = read_eval_results(cfg)
            if "mAP" not in eval_results:
                print(f"[warn] mAP not found in eval_results.txt for {run_folder.name} on {target_domain}")
                continue

            map_50_95 = round(eval_results["mAP"], 4)

            with open(RESULT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(meta + [map_50_95])
            existing_rows.add(row_key)


if __name__ == "__main__":
    main()
