#!/usr/bin/env python
"""Focused unit tests for the OWOD pseudo-unknown improvements (CPU, no data)."""
import os
import tempfile

import torch

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage


def make_heads(**over):
    kw = dict(
        num_classes=81,
        batch_size_per_image=10,
        positive_fraction=0.25,
        proposal_matcher=Matcher([0.5], [0, 1], allow_low_quality_matches=False),
        enable_thresold_autolabelling=False,
        unk_k=1,
        unk_select_enable=True,
        unk_select_all_proposals=True,
        unk_select_mode="quantile",
        unk_select_quantile=0.95,
        unk_select_max=3,
        unk_select_min_score=0.2,
        unk_select_nms_iou=0.5,
        unk_select_min_candidates=20,
        unk_select_gt_iou_max=0.3,
    )
    kw.update(over)
    return ROIHeads(**kw)


def boxes_from(xyxy):
    return Boxes(torch.tensor(xyxy, dtype=torch.float32))


def test_select_gt_suppression():
    heads = make_heads()
    # 40 background proposals in a row; proposal 0 overlaps GT almost perfectly.
    boxes = torch.tensor([[i * 20.0, 0, i * 20.0 + 100, 100] for i in range(40)])
    gt = boxes_from([[0, 0, 100, 100]])
    scores = torch.linspace(-3, 3, 40)
    scores[0] = 5.0  # would be selected without GT suppression
    scores[1] = 4.0  # IoU 0.6 with GT -> suppressed
    bg = torch.ones(40, dtype=torch.bool)
    keep, thr, suppressed = heads._select_unknown_candidates(
        scores, bg, boxes, gt, 3
    )
    assert suppressed >= 2, suppressed
    assert 0 not in keep.tolist() and 1 not in keep.tolist(), keep
    assert keep.numel() <= 3
    assert thr is not None and thr >= 0.2

    # delta=0 disables the filter
    heads0 = make_heads(unk_select_gt_iou_max=0.0)
    keep0, _, sup0 = heads0._select_unknown_candidates(scores, bg, boxes, gt, 3)
    assert sup0 == 0 and 0 in keep0.tolist(), (keep0, sup0)
    print("PASS gt suppression", keep.tolist(), round(thr, 3), suppressed)


def test_select_topk_and_nms():
    heads = make_heads(unk_select_mode="topk")
    boxes = torch.tensor([[0, 0, 100, 100]] * 10, dtype=torch.float32)
    scores = torch.arange(10, dtype=torch.float32)
    keep, thr, sup = heads._select_unknown_candidates(
        scores, torch.ones(10, dtype=torch.bool), boxes, None, 3
    )
    assert thr is None and keep.numel() == 3
    print("PASS topk fallback", keep.tolist())

    heads2 = make_heads(unk_select_min_candidates=3, unk_select_quantile=0.5)
    boxes2 = torch.tensor(
        [[0, 0, 100, 100], [1, 1, 101, 101], [200, 200, 300, 300],
         [400, 400, 500, 500], [600, 600, 700, 700], [800, 800, 900, 900]],
        dtype=torch.float32,
    )
    scores2 = torch.tensor([5.0, 4.9, 4.0, 3.0, 2.0, 1.0])
    keep2, thr2, _ = heads2._select_unknown_candidates(
        scores2, torch.ones(6, dtype=torch.bool), boxes2, None, 3
    )
    # first two overlap almost perfectly -> only the higher one kept; cap 3
    assert 0 in keep2.tolist() and 1 not in keep2.tolist(), keep2
    assert keep2.numel() <= 3
    print("PASS nms", keep2.tolist(), round(thr2, 3))


def test_sample_proposals_appends():
    heads = make_heads(unk_select_min_score=-10.0, unk_select_quantile=0.5)
    n = 60
    boxes = torch.tensor([[i * 15.0, 0, i * 15.0 + 90, 90] for i in range(n)])
    gt_classes = torch.tensor([0])
    matched_idxs = torch.zeros(n, dtype=torch.long)
    matched_labels = torch.zeros(n, dtype=torch.long)
    objectness = torch.linspace(-2, 4, n)
    sampled, labels = heads._sample_proposals(
        matched_idxs, matched_labels, gt_classes, objectness, boxes,
        boxes_from([[0, 0, 90, 90]]),
    )
    assert sampled.numel() > 10, sampled.numel()  # appended beyond the 10 sampled
    assert (labels == 80).sum().item() >= 1, labels
    assert sampled.unique().numel() == sampled.numel(), "duplicate indices"
    print("PASS sample/append", sampled.numel(), int((labels == 80).sum()))


def build_head():
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81
    cfg.OWOD.CLUSTERING.ITEMS_PER_CLASS = 20
    cfg.OWOD.CLUSTERING.START_ITER = 10
    cfg.OWOD.CLUSTERING.UPDATE_MU_ITER = 10
    cfg.OWOD.CLUSTERING.MOMENTUM = 0.99
    cfg.OWOD.CLUSTERING.MARGIN = 10.0
    cfg.OWOD.ENABLE_CLUSTERING = True
    cfg.OWOD.UNK_MEMORY.ENABLE = True
    cfg.OWOD.UNK_MEMORY.NUM_PROTOTYPES = 4
    cfg.OWOD.UNK_MEMORY.WEIGHT_MODE = "inv_freq"
    cfg.OWOD.UNK_MEMORY.ALPHA = 0.5
    cfg.OWOD.UNK_MEMORY.MIN_CLUSTER = 2
    cfg.OWOD.UNK_MEMORY.WEIGHT_CLIP = 2.0
    cfg.OWOD.UNK_MEMORY.NORMALIZE = True
    cfg.OWOD.UNK_MEMORY.WARMUP_ITERS = 10
    cfg.OWOD.UNK_MEMORY.KMEANS_ITERS = 10
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.CHECKPOINT_PERIOD = 50
    cfg.OUTPUT_DIR = tempfile.mkdtemp()
    cfg.OWOD.FEATURE_STORE_SAVE_PATH = "feature_store"
    kwargs = FastRCNNOutputLayers.from_config(cfg, ShapeSpec(channels=2048))
    return FastRCNNOutputLayers(**kwargs)


def test_unknown_memory_and_loss():
    torch.manual_seed(0)
    head = build_head()
    d = 2048
    # Two well-separated unknown modes in the queue.
    a = torch.randn(10, d) + 20.0
    b = torch.randn(10, d) - 20.0
    head.feature_store.add(torch.cat([a, b]), torch.full((20,), 80, dtype=torch.long))
    head.means[80] = None
    head._update_unknown_centers(log=False, kmeans=False)
    assert head.unknown_centers.shape[0] == 1, head.unknown_centers.shape
    head._update_unknown_centers(log=False, kmeans=True)
    assert 1 <= head.unknown_centers.shape[0] <= 4, head.unknown_centers.shape
    print("PASS centers", tuple(head.unknown_centers.shape))

    # 4 known + 4 unknown features.
    feats = torch.cat([
        torch.randn(4, d) * 0.1,
        torch.randn(2, d) + 20.0,
        torch.randn(2, d) - 20.0,
    ])
    for c in range(20):
        head.means[c] = torch.randn(d) * 0.1
    inst = Instances((100, 100))
    inst.gt_classes = torch.tensor([0, 1, 2, 3, 80, 80, 80, 80])
    with EventStorage(11) as storage:
        loss = head.clstr_loss_l2_cdist(feats, [inst])
        raw = storage.latest()
        scalars = {k: (v[0] if isinstance(v, tuple) else v) for k, v in raw.items()}
    assert torch.isfinite(loss) and loss.item() >= 0, loss
    assert "unk_memory/weight_mean" in scalars, scalars.keys()
    assert abs(scalars["unk_memory/weight_mean"] - 1.0) < 1e-4
    assert scalars["unk_memory/weight_max"] <= 2.0 + 1e-4
    print("PASS loss", float(loss), {k: round(v, 4) for k, v in scalars.items() if k.startswith("unk_memory/weight")})


def test_box_reg_exclusion():
    torch.manual_seed(0)
    n = 6
    inst = Instances((100, 100))
    inst.proposal_boxes = boxes_from([[i * 10, 0, i * 10 + 50, 50] for i in range(n)])
    inst.gt_boxes = boxes_from([[0, 0, 50, 50]] * n)
    inst.gt_classes = torch.tensor([0, 1, 2, 3, 80, 80])
    logits = torch.zeros(n, 82)
    deltas = torch.randn(n, 81 * 4)
    out_ex = FastRCNNOutputs(
        Box2BoxTransform([10, 10, 5, 5]),
        logits, deltas, [inst], list(range(20, 80)), True, 0.0, "smooth_l1",
    )
    out_in = FastRCNNOutputs(
        Box2BoxTransform([10, 10, 5, 5]),
        logits.clone(), deltas.clone(), [inst], list(range(20, 80)), False, 0.0, "smooth_l1",
    )
    l_ex = out_ex.box_reg_loss()
    l_in = out_in.box_reg_loss()
    assert l_ex.item() >= 0 and l_in.item() >= 0
    assert l_ex.item() <= l_in.item() + 1e-6
    print("PASS box_reg", float(l_ex), float(l_in))


if __name__ == "__main__":
    test_select_gt_suppression()
    test_select_topk_and_nms()
    test_sample_proposals_appends()
    test_unknown_memory_and_loss()
    test_box_reg_exclusion()
    print("ALL UNIT TESTS PASSED")
