#!/usr/bin/env python
"""Run a trained OWOD checkpoint on arbitrary images.

Example (Task4 model, all 80 classes):
  python tools/predict_owod.py \
    --config-file configs/OWOD/t4/t4_test.yaml \
    --weights weights/method_t4_160500.pth \
    --input demo.jpg --output output/predictions

The script runs detector inference only.  The energy/Weibull calibration used
for WI/A-OSE is an evaluation-time step and is not needed for ordinary image
detection; for exact WI/A-OSE reproduction use tools/train_net.py with the
matching energy_dist_*.pkl (see REPRODUCE.md).
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/OWOD/t4/t4_test.yaml")
    p.add_argument("--weights", default="weights/method_t4_160500.pth")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="output/predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def make_metadata(names, max_class):
    need = max(max_class + 1, len(names))
    names = list(names) + [f"class_{i}" for i in range(len(names), need)]
    if need > 80 and len(names) == need:
        names[80] = "unknown"
    meta = MetadataCatalog.get("__owod_predict")
    meta.set(thing_classes=names)
    return meta


def main():
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        files = [in_path]
    if not files:
        raise SystemExit(f"no images found: {args.input}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metadata from the test set, padded for the unknown class if needed
    test_sets = cfg.DATASETS.TEST
    base_meta = MetadataCatalog.get(test_sets[0] if test_sets else "__unused")
    base_names = list(base_meta.get("thing_classes", []))

    for path in files:
        img = read_image(str(path), format="BGR")
        pred = predictor(img)
        inst = pred["instances"].to("cpu")

        boxes = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else np.zeros((0, 4))
        scores = inst.scores.numpy() if inst.has("scores") else np.zeros((0,))
        classes = inst.pred_classes.numpy() if inst.has("pred_classes") else np.zeros((0,), dtype=np.int64)
        max_cls = int(classes.max()) if len(classes) else -1
        meta = make_metadata(base_names, max_cls)
        names = meta.get("thing_classes")

        records = []
        for box, score, cls in zip(boxes, scores, classes):
            c = int(cls)
            records.append({
                "class_id": c,
                "class_name": names[c] if c < len(names) else f"class_{c}",
                "score": float(score),
                "box_xyxy": [float(x) for x in box],
            })

        out_img = out_dir / f"{path.stem}_pred.jpg"
        vis = Visualizer(img[:, :, ::-1], meta, instance_mode=ColorMode.IMAGE)
        vis_out = vis.draw_instance_predictions(inst)
        vis_out.save(str(out_img))

        out_json = out_dir / f"{path.stem}_pred.json"
        out_json.write_text(json.dumps({
            "image": str(path),
            "checkpoint": str(args.weights),
            "detections": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"{path}: {len(records)} detections -> {out_img}")
        for r in records[:10]:
            print(f"  {r['class_name']:20s} {r['score']:.3f} {r['box_xyxy']}")


if __name__ == "__main__":
    main()
