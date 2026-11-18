#!/usr/bin/env python
"""True-class diagnostics for predicted unknown boxes.

The training pseudo-unknown queue has no true class labels, so its category
entropy is not identifiable. This script measures the *evaluated* unknown
predictions instead: it matches each predicted unknown box to the original
XML annotation (VOC/COCO names are preserved there) and reports

  * purity of predicted unknown boxes (true unknown / known / background)
  * true-class distribution and entropy of predicted unknown boxes
  * per-true-class recall at IoU 0.5

Usage:
  PYTHONPATH=. DETECTRON2_DATASETS=/data/OWOD/datasets/VOCdevkit \
  python tools/diagnose_unknown_predictions.py \
    --config-file configs/OWOD/t1/t1_test.yaml \
    --weights /data/OWOD/output_v3/t1_unk_final/model_final.pth \
    --dataset voc_coco_2007_test \
    --out /data/OWOD/output_v3/t1_unk_final/pred_class_report.json
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.data.datasets import builtin  # noqa: E402,F401  (registers datasets)
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator  # noqa: E402

# Task-1 known classes. The released XMLs may use either the VOC names or the
# COCOified names (the data preparation renames six VOC classes), so include both.
KNOWN_NAMES = {
    "aeroplane", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dining table", "dog", "horse",
    "motorbike", "motorcycle", "person", "pottedplant", "potted plant", "sheep",
    "sofa", "couch", "train", "tvmonitor", "tv",
}


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def parse_xml(path):
    tree = ET.parse(path)
    out = []
    for obj in tree.findall("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        box = [
            float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")
        ]
        out.append((name, box))
    return out


def entropy(counts):
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    if vals.sum() <= 0:
        return 0.0
    p = vals / vals.sum()
    return max(0.0, float(-(p * np.log(p + 1e-12)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/OWOD/t1/t1_test.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="voc_coco_2007_test")
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--energy-dir", default=None,
                    help="directory containing energy_dist_<seen>.pkl; default: weights dir")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all images")
    ap.add_argument("--start-index", type=int, default=0, help="skip the first N dataset entries")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--detections", type=int, default=50)
    ap.add_argument("--out", default="pred_class_report.json")
    args = ap.parse_args()

    if args.ann_dir is None:
        root = os.getenv("DETECTRON2_DATASETS", "datasets")
        args.ann_dir = os.path.join(root, "VOC2007", "Annotations")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections
    # The evaluator looks for energy_dist_<num_seen>.pkl under OUTPUT_DIR.
    cfg.OUTPUT_DIR = args.energy_dir or os.path.dirname(os.path.abspath(args.weights))
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    # Use the exact same energy relabelling as the official evaluator so that
    # "predicted unknown" here means the same thing as in U-Recall/A-OSE/WI.
    evaluator = PascalVOCDetectionEvaluator(args.dataset, cfg)
    if not getattr(evaluator, "energy_distribution_loaded", False):
        print("[warn] energy distribution not found under", cfg.OUTPUT_DIR,
              "- falling back to raw class-80 predictions")

    dataset = DatasetCatalog.get(args.dataset)
    if args.start_index and args.start_index > 0:
        dataset = dataset[args.start_index :]
    if args.max_images and args.max_images > 0:
        dataset = dataset[: args.max_images]

    total_pred_unknown = 0
    true_unknown = 0
    known_fp = 0
    background_fp = 0
    pred_class_counts = {}
    gt_class_counts = {}
    skipped_xml = 0

    for i, d in enumerate(dataset):
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        out = predictor(img)["instances"].to("cpu")
        boxes = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        classes = out.pred_classes
        if out.has("logits"):
            classes = evaluator.update_label_based_on_energy(out.logits, classes.clone())
        classes = classes.numpy()

        xml_path = os.path.join(args.ann_dir, str(d["image_id"]) + ".xml")
        if os.path.isfile(xml_path):
            gt = parse_xml(xml_path)
        else:
            gt = []
            skipped_xml += 1

        # denominator for per-class recall
        for name, _ in gt:
            if name not in KNOWN_NAMES:
                gt_class_counts[name] = gt_class_counts.get(name, 0) + 1

        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) != 80 or float(score) < args.score_thresh:
                continue
            total_pred_unknown += 1
            best_iou, best_name = 0.0, None
            for name, gbox in gt:
                v = iou(box, gbox)
                if v > best_iou:
                    best_iou, best_name = v, name
            if best_iou >= args.iou_thresh and best_name is not None:
                if best_name in KNOWN_NAMES:
                    known_fp += 1
                else:
                    true_unknown += 1
                    pred_class_counts[best_name] = pred_class_counts.get(best_name, 0) + 1
            else:
                background_fp += 1

        if (i + 1) % 500 == 0:
            print("processed", i + 1, "/", len(dataset))

    n_classes = len(pred_class_counts)
    ent = entropy(pred_class_counts)
    report = {
        "dataset": args.dataset,
        "num_images": len(dataset),
        "skipped_xml": skipped_xml,
        "num_pred_unknown": total_pred_unknown,
        "num_true_unknown": true_unknown,
        "num_known_fp": known_fp,
        "num_background_fp": background_fp,
        "purity_true_unknown": true_unknown / max(total_pred_unknown, 1),
        "pred_class_counts": pred_class_counts,
        "gt_class_counts": gt_class_counts,
        "true_class_entropy": ent,
        "normalized_true_class_entropy": ent / np.log(max(n_classes, 2)),
        "per_class_recall": {
            k: pred_class_counts.get(k, 0) / max(v, 1)
            for k, v in sorted(gt_class_counts.items())
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
