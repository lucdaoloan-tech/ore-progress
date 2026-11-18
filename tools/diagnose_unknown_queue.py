#!/usr/bin/env python
"""Diagnose the unknown-feature queue used by ORE / OReg.

Usage:
  PYTHONPATH=. python tools/diagnose_unknown_queue.py \
      --feature-store D:/OWOD_data/output/t1/feature_store/feat.pt \
      --energy-dir D:/OWOD_data/output/t1/energy \
      --out D:/OWOD_data/output/t1/unknown_queue_report.json

The script reports:
  * queue size and feature norm statistics
  * pairwise cosine similarity (mean / max)
  * k-means cluster sizes and cluster-size entropy
  * optional known-vs-unknown energy AUROC from a directory of energy .pt files
  * optional training scalars from a TensorBoard event file/dir
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

# Allow running from the repository root without setting PYTHONPATH.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from detectron2.utils.store import Store  # noqa: E402  (needed to unpickle feat.pt)


def _to_matrix(items):
    rows = []
    for item in items:
        if torch.is_tensor(item):
            arr = item.detach().cpu().numpy().reshape(-1)
        elif isinstance(item, (list, tuple)):
            # Store.retrieve(-1) converts each feature vector into a list of 0-dim tensors.
            parts = [
                t.detach().cpu() if torch.is_tensor(t) else torch.tensor(t)
                for t in item
            ]
            arr = torch.stack(parts).numpy().reshape(-1)
        else:
            arr = np.asarray(item, dtype=np.float32).reshape(-1)
        rows.append(arr.astype(np.float32))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0)


def _kmeans_numpy(x, k, iters=20):
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, x.shape[1]), dtype=np.float32)
    if n <= k:
        labels = np.arange(n, dtype=np.int64)
        return labels, x.copy()

    # farthest-point initialization
    mean = x.mean(axis=0, keepdims=True)
    centers = [x[int(np.argmax(np.linalg.norm(x - mean, axis=1)))]]
    for _ in range(1, k):
        d = np.min(
            np.linalg.norm(x[:, None, :] - np.stack(centers)[None, :, :], axis=2), axis=1
        )
        centers.append(x[int(np.argmax(d))])
    centers = np.stack(centers, axis=0)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(int(iters), 1)):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centers = []
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers.append(x[mask].mean(axis=0))
            else:
                new_centers.append(centers[j])
        centers = np.stack(new_centers, axis=0)
    return labels, centers


def _auroc(pos, neg):
    """AUROC with average ranks; higher score means more likely positive."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0  # 1-based average rank
        i = j + 1
    n_pos = pos.size
    n_neg = neg.size
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _energy_stats(energy_dir, unknown_index, num_seen_classes, temperature=1.5):
    files = sorted(
        glob.glob(os.path.join(energy_dir, "*.pt"))
        + glob.glob(os.path.join(energy_dir, "*.pkl"))
    )
    known, unknown = [], []
    for path in files:
        try:
            logits, classes = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if logits.numel() == 0:
            continue
        # Same free energy as Weibull fitting / evaluation:
        # E = T * logsumexp(z[:num_seen] / T). Higher E means more unknown.
        energy = (
            float(temperature)
            * torch.logsumexp(logits[:, :num_seen_classes] / float(temperature), dim=1)
        ).detach().cpu().numpy()
        cls = classes.detach().cpu().numpy() if torch.is_tensor(classes) else np.asarray(classes)
        for e, c in zip(energy, cls):
            if c == logits.shape[1] - 1:  # background
                continue
            # Match the evaluator: every class outside the seen range is unknown.
            if c >= num_seen_classes:
                unknown.append(float(e))
            else:
                known.append(float(e))
    return {
        "num_energy_files": len(files),
        "num_known": len(known),
        "num_unknown": len(unknown),
        "energy_auroc": _auroc(unknown, known),
    }


def _event_stats(events_path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    files = [events_path]
    if os.path.isdir(events_path):
        files = sorted(glob.glob(os.path.join(events_path, "events.out.tfevents.*")))
    out = {}
    for path in files:
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag.startswith("unk_select/") or tag.startswith("unk_memory/"):
                vals = ea.Scalars(tag)
                if vals:
                    out[tag] = {
                        "last": float(vals[-1].value),
                        "mean": float(np.mean([v.value for v in vals])),
                        "steps": int(vals[-1].step),
                    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", required=True, help="path to feature_store/feat.pt")
    parser.add_argument("--unknown-index", type=int, default=80)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--energy-dir", default=None)
    parser.add_argument("--num-seen-classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--events", default=None)
    parser.add_argument("--out", default="unknown_queue_report.json")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    store = torch.load(args.feature_store, map_location="cpu", weights_only=False)
    if not isinstance(store, Store):
        raise TypeError("feature store is not a detectron2 Store: %s" % type(store))
    items = store.retrieve(-1)[args.unknown_index]
    x = _to_matrix(items)

    report = {
        "feature_store": args.feature_store,
        "unknown_index": args.unknown_index,
        "queue_size": int(x.shape[0]),
        "feature_dim": int(x.shape[1]) if x.ndim == 2 else 0,
    }
    if x.shape[0] == 0:
        report["note"] = "unknown queue is empty"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return

    norms = np.linalg.norm(x, axis=1)
    report["norm_mean"] = float(norms.mean())
    report["norm_std"] = float(norms.std())

    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = xn @ xn.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)] if sim.shape[0] > 1 else np.array([])
    report["mean_pairwise_cosine"] = float(off.mean()) if off.size else None
    report["max_pairwise_cosine"] = float(off.max()) if off.size else None
    report["min_pairwise_cosine"] = float(off.min()) if off.size else None

    labels, centers = _kmeans_numpy(x, args.k, iters=20)
    sizes = np.bincount(labels, minlength=args.k).astype(int).tolist()
    probs = np.asarray(sizes, dtype=np.float64)
    probs = probs / max(probs.sum(), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    report["cluster_sizes"] = sizes
    report["cluster_entropy"] = entropy
    report["normalized_cluster_entropy"] = float(entropy / np.log(max(args.k, 2)))
    report["empty_clusters"] = int(sum(1 for s in sizes if s == 0))
    report["kmeans_degenerate_n_le_k"] = bool(x.shape[0] <= args.k)

    if args.energy_dir:
        report.update(_energy_stats(
            args.energy_dir, args.unknown_index, args.num_seen_classes, args.temperature
        ))
    if args.events:
        report["training_scalars"] = _event_stats(args.events)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].bar(np.arange(len(sizes)), sizes)
            axes[0].set_title("unknown sub-prototype sizes")
            axes[0].set_xlabel("cluster")
            axes[1].hist(norms, bins=min(20, max(3, x.shape[0])))
            axes[1].set_title("feature norm")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
        except Exception as exc:
            report["plot_error"] = str(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
