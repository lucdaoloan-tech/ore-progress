#!/usr/bin/env python
import torch
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage

heads = ROIHeads(
    num_classes=81, batch_size_per_image=10, positive_fraction=0.25,
    proposal_matcher=Matcher([0.5], [0, 1], allow_low_quality_matches=False),
    enable_thresold_autolabelling=False, unk_k=1,
    unk_select_enable=True, unk_select_all_proposals=True,
    unk_select_mode="quantile", unk_select_quantile=0.5,
    unk_select_max=3, unk_select_min_score=-10.0,
    unk_select_nms_iou=0.5, unk_select_min_candidates=3,
    unk_select_gt_iou_max=0.3, num_seen_classes=40,
)
proposals = Instances((200, 200))
boxes = torch.tensor([[i * 10.0, 0, i * 10.0 + 40, 40] for i in range(30)])
proposals.proposal_boxes = Boxes(boxes)
proposals.objectness_logits = torch.linspace(-2, 4, 30)
targets = Instances((200, 200))
targets.gt_boxes = Boxes(torch.tensor([[0.0, 0, 40, 40], [100.0, 0, 140, 40]]))
targets.gt_classes = torch.tensor([5, 45])  # 5 known, 45 future
with EventStorage(0):
    out = heads.label_and_sample_proposals([proposals], [targets])[0]
labels = out.gt_classes
print("labels", labels.tolist())
print("future_label_count", int((labels == 45).sum()))
print("unknown_count", int((labels == 80).sum()))
print("bg_count", int((labels == 81).sum()))
assert int((labels == 45).sum()) == 0, "future class must not be a foreground label"
print("FUTURE-CLASS TEST PASSED")
