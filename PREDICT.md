# Run a trained checkpoint on your own image

The trained checkpoints have been copied to:

```text
OWOD-master-progress/weights/
```

Weights are intentionally ignored by Git (`.gitignore`); do not commit them.
The same files are also available as GitHub Release `v1.0`.

## 1. Choose a checkpoint

For a general image with COCO/VOC objects, use the **Task4** model,
because it has all 80 classes:

```text
weights/method_t4_160500.pth
configs/OWOD/t4/t4_test.yaml
```

For Task1/2/3, use the corresponding config and checkpoint:

```text
weights/method_t1_18k.pth      configs/OWOD/t1/t1_test.yaml
weights/method_t2_74k.pth      configs/OWOD/t2/t2_test.yaml
weights/method_t3_116500.pth   configs/OWOD/t3/t3_test.yaml
```

Task1's `model_final.pth` is not used; use the 18k checkpoint above.

## 2. Single image

From the repository root:

```bash
python tools/predict_owod.py \
  --config-file configs/OWOD/t4/t4_test.yaml \
  --weights weights/method_t4_160500.pth \
  --input "C:/path/to/your_image.jpg" \
  --output output/my_predictions \
  --threshold 0.5
```

Windows PowerShell (single line):

```powershell
python tools/predict_owod.py --config-file configs/OWOD/t4/t4_test.yaml --weights weights/method_t4_160500.pth --input "C:\path\to\image.jpg" --output output\my_predictions --threshold 0.5
```

Outputs:

```text
output/my_predictions/<image>_pred.jpg    annotated image
output/my_predictions/<image>_pred.json   boxes, scores, class IDs/names
```

## 3. A directory of images

```bash
python tools/predict_owod.py \
  --config-file configs/OWOD/t4/t4_test.yaml \
  --weights weights/method_t4_160500.pth \
  --input "C:/path/to/images" \
  --output output/my_predictions
```

## 4. Note about WI / A-OSE

This script does **ordinary detector inference**.  The energy/Weibull
calibration used for WI/A-OSE is an evaluation step and is not applied here.
For exact WI/A-OSE reproduction, use `tools/train_net.py --eval-only` with the
matching `energy_dist_*.pkl` as described in `REPRODUCE.md`.

Task4 has all 80 classes introduced, so it has no unknown GT; for unknown
detection use Task1/2/3 checkpoints (the unknown class is index 80 in the model).
