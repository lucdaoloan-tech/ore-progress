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
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/OWOD/t4/t4_test.yaml")
    p.add_argument("--weights", default="weights/method_t4_160500.pth")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="output/predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--font-size", type=int, default=14)
    p.add_argument("--tag", default="", help="optional prefix suffix: source/method")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def load_font(size):
    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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
        need = max(max_cls + 1, len(base_names))
        names = list(base_names) + [f"class_{i}" for i in range(len(base_names), need)]
        if need > 80:
            names[80] = "unknown"

        records = []
        for box, score, cls in zip(boxes, scores, classes):
            c = int(cls)
            records.append({
                "class_id": c,
                "class_name": names[c] if c < len(names) else f"class_{c}",
                "score": float(score),
                "box_xyxy": [float(x) for x in box],
            })

        suffix = f"_{args.tag}" if args.tag else ""
        out_img = out_dir / f"{path.stem}{suffix}_pred.jpg"
        out_json = out_dir / f"{path.stem}{suffix}_pred.json"

        rgb = np.ascontiguousarray(img[:, :, ::-1])
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = load_font(args.font_size)
        for r in records:
            c = r["class_id"]
            x1, y1, x2, y2 = r["box_xyxy"]
            color = (210, 50, 40) if c >= 80 else (0, 110, 200)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=args.line_width)
            label = f'{r["class_name"]} {r["score"]:.2f}'
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 2)
            draw.rectangle((x1, ty, x1 + tw + 4, ty + th + 2), fill=color)
            draw.text((x1 + 2, ty), label, fill="white", font=font)
        pil.save(out_img, quality=92)

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
