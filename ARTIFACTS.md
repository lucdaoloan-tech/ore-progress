# Checkpoints and Evaluation Artifacts

The code repository intentionally does **not** contain datasets or `.pth`
checkpoints.  The final checkpoints are attached to the GitHub Release (or an
external artifact store) and are listed here for reproducibility.

## Final checkpoints (2× RTX 4090, batch 8, LR 0.01, seed 42)

| Task | Improved method | Official source baseline | Notes |
|---|---|---|---|
| Task1 | `method_t1_18k.pth` | `source_t1_18k.pth` | use iteration-17999 model, **not** the val-overwritten `model_final.pth` |
| Task2 | `method_t2_74k.pth` | `source_t2_74k.pth` | after `t2_ft` |
| Task3 | `method_t3_116500.pth` | `source_t3_116500.pth` | after `t3_ft` |
| Task4 | `method_t4_160500.pth` | `source_t4_160500.pth` | after `t4_ft` |

All eight files are 258 MB full checkpoints (model + iteration, no need to
resume training for evaluation).  SHA256 checksums are in
`SHA256SUMS.txt` next to the release assets.

The consolidated copy on the training server is:

```text
/root/autodl-tmp/OWOD_artifacts/weights/
```

The original locations are:

```text
# improved method
/root/autodl-tmp/OWOD/output/t1_unk/model_0017999.pth
/root/autodl-tmp/OWOD/output/t2_ft/model_final.pth
/root/autodl-tmp/OWOD/output/t3_ft/model_final.pth
/root/autodl-tmp/OWOD/output/t4_ft/model_final.pth

# official source baseline
/root/autodl-tmp/OWOD_src/output/t1_src/model_0017999.pth
/root/autodl-tmp/OWOD_src/output/t2_src_ft/model_final.pth
/root/autodl-tmp/OWOD_src/output/t3_src_ft/model_final.pth
/root/autodl-tmp/OWOD_src/output/t4_src_ft/model_final.pth
```

## Energy calibration files

The energy/Weibull calibration is tiny (1.4 KB each) and is needed to
reproduce the WI/A-OSE numbers.  It is kept with the evaluation outputs:

| Task | Improved method | Official source |
|---|---|---|
| Task1 | `output/t1_unk_eval/energy_dist_20.pkl` | `output/t1_src_eval/energy_dist_20.pkl` |
| Task2 | `output/t2_ft_eval/energy_dist_40.pkl` | `output/t2_src_ft_eval/energy_dist_40.pkl` |
| Task3 | `output/t3_ft_eval/energy_dist_60.pkl` | `output/t3_src_ft_eval/energy_dist_60.pkl` |
| Task4 | `output/t4_ft_eval/energy_dist_80.pkl` | `output/t4_src_ft_eval/energy_dist_80.pkl` |

## What to attach where

- **Git repository**: code, configs, tests, `IMPROVEMENTS.md`, this file.
- **GitHub Release assets**: the eight `.pth` files and `SHA256SUMS.txt`
  (or only the four `method_*.pth` files if the official-source weights are
  not needed; the source code itself is the official OWOD repository).
- **Optional archive**: evaluation logs, `energy_dist_*.pkl`, and the four-task
  comparison tables/JSON; these are small and can go in a separate `results.zip`.

Do **not** `git add` the `.pth` files: each is 258 MB, above GitHub's 100 MB
per-file limit.  Use a GitHub Release, Git LFS, or an external archive
(Zenodo / institutional storage / cloud drive) and link it from the README.
