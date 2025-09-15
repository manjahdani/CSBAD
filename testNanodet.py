import os, datetime, torch, pytorch_lightning as pl
from pathlib import Path
from nanodet.util import load_config, cfg, mkdir, NanoDetLightningLogger, convert_old_model
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask

# ------------------------------------------------------------------ paths
CONFIG_PATH = Path("/home/dani/CSBAD/nanodet/config/nanodet-plus-m_416.yml")
CKPT_PATH   = Path("/home/dani/Downloads/nanodet-plus-m_416_checkpoint.ckpt")
EVAL_TMP    = Path("./eval_tmp");  EVAL_TMP.mkdir(exist_ok=True)

# ------------------------------------------------------------------ helper
def read_map(save_dir):
    fp = Path(save_dir) / "eval_results.txt"
    if not fp.exists(): return None
    for ln in fp.read_text().splitlines():
        if "mAP" in ln:
            try: return float(ln.split(":")[1].strip())
            except: pass
    return None

# ------------------------------------------------------------------ evaluator patch
def patch_class_agnostic(evaluator):
    """
    Ignore the model's class ID and always report the single GT category.
    """
    gt_cat_id = evaluator.cat_ids[0]          # your only category (e.g. 1)
    def to_json(results):
        out = []
        for r in results:
            if not isinstance(r, dict):                 # skip ints etc.
                continue
            boxes, scores, img_id = r["bboxes"], r["scores"], r["image_id"]
            for i in range(len(boxes)):
                out.append({
                    "image_id":   img_id,
                    "category_id": gt_cat_id,           # ← all boxes = 1 class
                    "bbox":  [ round(x,2) for x in boxes[i].tolist() ],
                    "score": round(scores[i].item(), 5)
                })
        evaluator.results = out
        return out
    evaluator.results2json = to_json

# ------------------------------------------------------------------ main
def main():
    cams, total = 0, 0.0
    for cam in range(1, 10):
        tgt = f"cam{cam}"
        test_dir = Path(f"/home/dani/data/WALT-challenge/{tgt}/test")
        ann, img = test_dir/"labels_coco.json", test_dir/"images"
        if not ann.exists(): print(f"[skip] {tgt}"); continue

        print(f"→ Evaluating COCO model on {tgt}")
        load_config(cfg, str(CONFIG_PATH))          # keep original 80-class head
        cfg.defrost()
        cfg.data.val.update({"ann_path": str(ann), "img_path": str(img)})
        cfg.test_mode = "val"
        cfg.save_dir  = str(EVAL_TMP / f"{tgt}_{datetime.datetime.now():%Y%m%d%H%M%S}")
        mkdir(-1, Path(cfg.save_dir))

        logger = NanoDetLightningLogger(cfg.save_dir)
        val_ds  = build_dataset(cfg.data.val, "val")
        val_dl  = torch.utils.data.DataLoader(
                    val_ds, batch_size=cfg.device.batchsize_per_gpu, shuffle=False,
                    num_workers=cfg.device.workers_per_gpu, pin_memory=True,
                    collate_fn=naive_collate, drop_last=False)

        evaluator = build_evaluator(cfg.evaluator, val_ds)
        patch_class_agnostic(evaluator)             # *** the key line ***

        task = TrainingTask(cfg, evaluator)
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
        if "pytorch-lightning_version" not in ckpt:
            ckpt = convert_old_model(ckpt)
        task.load_state_dict(ckpt["state_dict"])

        acc = "gpu" if cfg.device.gpu_ids != -1 else "cpu"
        dev = cfg.device.gpu_ids if acc == "gpu" else None
        pl.Trainer(
            default_root_dir=cfg.save_dir, accelerator=acc, devices=dev,
            logger=logger, log_every_n_steps=cfg.log.interval,
            num_sanity_val_steps=0,
        ).test(task, val_dl)

        mp = read_map(cfg.save_dir)
        if mp is None: print("  mAP missing"); continue
        print(f"  → mAP50-95: {mp}")
        total += mp;  cams += 1

    if cams:
        print(f"\n✅ Avg mAP50-95 over {cams} cams: {round(total/cams,4)}")
    else:
        print("❌ No cameras evaluated")

if __name__ == "__main__":
    main()
