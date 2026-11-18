# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, Union
import torch
import os
import math
import shortuuid
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.store import Store

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, predictions, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, prediction
        )
        for scores_per_image, boxes_per_image, image_shape, prediction in zip(scores, boxes, image_shapes, predictions)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, prediction
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    logits = prediction
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        logits = logits[valid_mask]

    scores = scores[:, :-1]
    logits = logits[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    logits = logits[filter_inds[:,0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    logits = logits[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.logits = logits
    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        invalid_class_range,
        box_reg_exclude_unknown=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]
        self.invalid_class_range = invalid_class_range
        self.box_reg_exclude_unknown = box_reg_exclude_unknown

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            self.pred_class_logits[:, self.invalid_class_range] = -10e10
            # self.log_logits(self.pred_class_logits, self.gt_classes)
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def log_logits(self, logits, cls):
        data = (logits, cls)
        location = '/home/fk1/workspace/OWOD/output/logits/' + shortuuid.uuid() + '.pkl'
        torch.save(data, location)

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_mask = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        if self.box_reg_exclude_unknown:
            fg_mask = fg_mask & (self.gt_classes != bg_class_ind - 1)
        fg_inds = nonzero_tuple(fg_mask)[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}#self.softmax_cross_entropy_loss()是将未知类已知

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

class AE(nn.Module):
    def __init__(self, input_size, z_dim):
        super(AE,self).__init__()
        self.e1 = nn.Linear(input_size,z_dim)
        self.d1 = nn.Linear(z_dim, input_size)

    def encoder(self, x):
        z = self.e1(x)
        z = torch.relu(z)
        return z

    def decoder(self, z):
        x = self.d1(z)
        x = torch.relu(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        clustering_items_per_class,
        clustering_start_iter,
        clustering_update_mu_iter,
        clustering_momentum,
        clustering_z_dimension,
        enable_clustering,
        prev_intro_cls,
        curr_intro_cls,
        max_iterations,
        checkpoint_period=5000,
        box_reg_exclude_unknown=False,
        output_dir,
        feat_store_path,
        margin,
        unk_memory_enable=False,
        unk_num_prototypes=4,
        unk_weight_mode="inv_freq",
        unk_alpha=0.5,
        unk_min_cluster=2,
        unk_weight_clip=2.0,
        unk_normalize=True,
        unk_warmup_iters=3000,
        unk_kmeans_iters=10,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.num_classes = num_classes
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.clustering_momentum = clustering_momentum

        self.hingeloss = nn.HingeEmbeddingLoss(2)
        self.enable_clustering = enable_clustering

        self.prev_intro_cls = prev_intro_cls
        self.curr_intro_cls = curr_intro_cls
        self.seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.invalid_class_range = list(range(self.seen_classes, self.num_classes-1))
        logging.getLogger(__name__).info("Invalid class range: " + str(self.invalid_class_range))

        self.max_iterations = max_iterations
        self.checkpoint_period = max(1, int(checkpoint_period))
        self.box_reg_exclude_unknown = box_reg_exclude_unknown
        self.feature_store_is_stored = False
        self.output_dir = output_dir
        self.feat_store_path = feat_store_path
        self.feature_store_save_loc = os.path.join(self.output_dir, self.feat_store_path, 'feat.pt')
        os.makedirs(os.path.dirname(self.feature_store_save_loc), exist_ok=True)

        if os.path.isfile(self.feature_store_save_loc):
            logging.getLogger(__name__).info('Trying to load feature store from ' + self.feature_store_save_loc)
            self.feature_store = torch.load(self.feature_store_save_loc, weights_only=False)
        else:
            logging.getLogger(__name__).info('Feature store not found in ' +
                                             self.feature_store_save_loc + '. Creating new feature store.')
            self.feature_store = Store(num_classes + 1, clustering_items_per_class)
        self.means = [None for _ in range(num_classes + 1)]
        self.margin = margin

        # Unknown memory: K sub-prototypes + frequency/novelty weighting.
        self.unk_memory_enable = unk_memory_enable
        self.unk_num_prototypes = unk_num_prototypes
        self.unk_weight_mode = unk_weight_mode
        self.unk_alpha = unk_alpha
        self.unk_min_cluster = unk_min_cluster
        self.unk_weight_clip = unk_weight_clip
        self.unk_normalize = unk_normalize
        self.unk_warmup_iters = unk_warmup_iters
        self.unk_kmeans_iters = unk_kmeans_iters
        self.unknown_idx = self.num_classes - 1
        self.unknown_centers = None
        self.hingeloss_none = nn.HingeEmbeddingLoss(2, reduction="none")

        # self.ae_model = AE(input_size, clustering_z_dimension)
        # self.ae_model.apply(Xavier)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT, "loss_clustering": 0.1},
            "clustering_items_per_class" : cfg.OWOD.CLUSTERING.ITEMS_PER_CLASS,
            "clustering_start_iter" : cfg.OWOD.CLUSTERING.START_ITER,
            "clustering_update_mu_iter" : cfg.OWOD.CLUSTERING.UPDATE_MU_ITER,
            "clustering_momentum"   : cfg.OWOD.CLUSTERING.MOMENTUM,
            "clustering_z_dimension": cfg.OWOD.CLUSTERING.Z_DIMENSION,
            "enable_clustering"     : cfg.OWOD.ENABLE_CLUSTERING,
            "prev_intro_cls"        : cfg.OWOD.PREV_INTRODUCED_CLS,
            "curr_intro_cls"        : cfg.OWOD.CUR_INTRODUCED_CLS,
            "max_iterations"        : cfg.SOLVER.MAX_ITER,
            "checkpoint_period"     : cfg.SOLVER.CHECKPOINT_PERIOD,
            "box_reg_exclude_unknown": cfg.OWOD.BOX_REG_EXCLUDE_UNKNOWN,
            "output_dir"            : cfg.OUTPUT_DIR,
            "feat_store_path"       : cfg.OWOD.FEATURE_STORE_SAVE_PATH,
            "margin"                : cfg.OWOD.CLUSTERING.MARGIN,
            "unk_memory_enable"     : cfg.OWOD.UNK_MEMORY.ENABLE,
            "unk_num_prototypes"    : cfg.OWOD.UNK_MEMORY.NUM_PROTOTYPES,
            "unk_weight_mode"       : cfg.OWOD.UNK_MEMORY.WEIGHT_MODE,
            "unk_alpha"             : cfg.OWOD.UNK_MEMORY.ALPHA,
            "unk_min_cluster"       : cfg.OWOD.UNK_MEMORY.MIN_CLUSTER,
            "unk_weight_clip"       : cfg.OWOD.UNK_MEMORY.WEIGHT_CLIP,
            "unk_normalize"         : cfg.OWOD.UNK_MEMORY.NORMALIZE,
            "unk_warmup_iters"      : cfg.OWOD.UNK_MEMORY.WARMUP_ITERS,
            "unk_kmeans_iters"      : cfg.OWOD.UNK_MEMORY.KMEANS_ITERS,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def update_feature_store(self, features, proposals):
        # cat(..., dim=0) concatenates over all images in the batch
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        self.feature_store.add(features, gt_classes)

        storage = get_event_storage()

        # 定期保存 feature_store，保证 --resume 时 means/unknown_centers 能重建
        if (storage.iter + 1) % self.checkpoint_period == 0 and comm.is_main_process():
            logging.getLogger(__name__).info(
                'Saving image store at iteration ' + str(storage.iter + 1) + ' to ' + self.feature_store_save_loc
            )
            torch.save(self.feature_store, self.feature_store_save_loc)

        if storage.iter == self.max_iterations-1 and self.feature_store_is_stored is False and comm.is_main_process():
            logging.getLogger(__name__).info('Saving image store at iteration ' + str(storage.iter) + ' to ' + self.feature_store_save_loc)
            torch.save(self.feature_store, self.feature_store_save_loc)
            self.feature_store_is_stored = True

        # self.feature_store.add(F.normalize(features, dim=0), gt_classes)
        # self.feature_store.add(self.ae_model.encoder(features), gt_classes)


    def _init_means_from_store(self):
        """从 feature_store 重建类均值和 unknown 子原型，供 --resume 使用。"""
        items = self.feature_store.retrieve(-1)
        for index, item in enumerate(items):
            if len(item) == 0:
                self.means[index] = None
            else:
                self.means[index] = torch.tensor(item).mean(dim=0)
        self._update_unknown_centers(log=False, kmeans=False)

    def _update_unknown_centers(self, log=True, kmeans=True):
        """把 unknown 队列聚成 K 个子原型，并更新 self.unknown_centers。

        Args:
            kmeans: True  -> farthest-point k-means, K=min(NUM_PROTOTYPES, queue size)
                    False -> single prototype = queue mean. The doc initializes the
                    unknown class with the mean at CLUSTERING.START_ITER and only
                    runs k-means on later UPDATE_MU_ITER boundaries.
        """
        if not self.unk_memory_enable:
            return
        # Store.retrieve(单个类) 会把特征向量 flatten；这里用 retrieve(-1) 取“每类的向量列表”
        items = self.feature_store.retrieve(-1)[self.unknown_idx]
        if len(items) == 0:
            return

        feats = []
        for item in items:
            if torch.is_tensor(item):
                feats.append(item.detach().float())
            else:
                feats.append(torch.tensor(item, dtype=torch.float))
        feats = torch.stack(feats)

        # 子原型保存在 CPU，避免和原实现里 CPU 的 self.means 混用导致 device mismatch；
        # 计算 loss 时再搬到特征所在设备。
        if not kmeans:
            centers = feats.mean(dim=0, keepdim=True).detach().cpu()
        else:
            k = min(max(int(self.unk_num_prototypes), 1), feats.shape[0])
            centers = self._kmeans(feats, k, self.unk_kmeans_iters).detach().cpu()

            # 丢掉空簇：k-means 在样本很少/重复时可能留下没有分配点的中心
            if centers.shape[0] > 1:
                assign = torch.cdist(feats.detach().cpu().float(), centers).argmin(dim=1)
                counts = torch.bincount(assign, minlength=centers.shape[0])
                nonempty = counts > 0
                if nonempty.any():
                    centers = centers[nonempty]

        # 队列级簇大小（不是当前 batch 的覆盖情况）
        if log:
            storage = get_event_storage()
            queue_assign = torch.cdist(feats.detach().cpu().float(), centers).argmin(dim=1)
            queue_counts = torch.bincount(queue_assign, minlength=centers.shape[0])
            storage.put_scalar("unk_memory/queue_min_cluster", float(queue_counts.min().item()))
            storage.put_scalar("unk_memory/queue_max_cluster", float(queue_counts.max().item()))

        if self.unknown_centers is not None and self.unknown_centers.shape == centers.shape:
            self.unknown_centers = self.clustering_momentum * self.unknown_centers + \
                                   (1 - self.clustering_momentum) * centers
        else:
            self.unknown_centers = centers
        # 保持 self.means[unknown] 可用，兼容原来的标签矩阵/其他代码路径
        self.means[self.unknown_idx] = self.unknown_centers.mean(dim=0)
        if log:
            storage = get_event_storage()
            storage.put_scalar("unk_memory/num_centers", int(self.unknown_centers.shape[0]))

    @staticmethod
    def _kmeans(x, k, iters):
        """极简 k-means：farthest-point 初始化 + 若干轮 assign/update。"""
        n = x.shape[0]
        if n <= k:
            return x.clone()
        mean = x.mean(dim=0, keepdim=True)
        centers = [x[torch.argmax(torch.cdist(x, mean).squeeze(1))]]
        for _ in range(1, k):
            d = torch.cdist(x, torch.stack(centers)).min(dim=1).values
            centers.append(x[torch.argmax(d)])
        centers = torch.stack(centers)

        for _ in range(max(int(iters), 1)):
            assign = torch.cdist(x, centers).argmin(dim=1)
            new_centers = []
            for j in range(k):
                m = assign == j
                if m.any():
                    new_centers.append(x[m].mean(dim=0))
                else:
                    if new_centers:
                        d = torch.cdist(x, torch.stack(new_centers)).min(dim=1).values
                    else:
                        d = torch.cdist(x, x.mean(dim=0, keepdim=True)).squeeze(1)
                    new_centers.append(x[torch.argmax(d)])
            centers = torch.stack(new_centers)
        return centers

    def clstr_loss_l2_cdist(self, input_features, proposals):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        mask = gt_classes != self.num_classes
        fg_features = input_features[mask]
        classes = gt_classes[mask]
        if fg_features.numel() == 0:
            return torch.tensor(0.0, device=input_features.device)

        # unknown 那一列先用子原型均值占位，保持 82 列标签矩阵不变。
        # Copy the list so that classes whose queue is still empty stay None
        # (doc: "原型保持为空，直到首次出现样本"); for the distance matrix they
        # are represented by a zero placeholder, and their target stays 0, which
        # HingeEmbeddingLoss treats like -1 exactly as ORE did.
        all_means = list(self.means)
        if self.unk_memory_enable and self.unknown_centers is not None:
            all_means[self.unknown_idx] = self.unknown_centers.mean(dim=0)

        length = None
        for item in all_means:
            if item is not None:
                length = item.shape
                break
        if length is None:
            return torch.tensor(0.0, device=fg_features.device)

        for i, item in enumerate(all_means):
            if item is None:
                all_means[i] = torch.zeros(length)

        # mean 都在 CPU，计算前统一搬到当前特征所在设备
        all_means = [m.to(fg_features.device) for m in all_means]
        distances = torch.cdist(fg_features, torch.stack(all_means), p=self.margin)

        unk_mask = classes == self.unknown_idx
        if self.unk_memory_enable and self.unknown_centers is not None:
            # unknown 特征拉向“最近子原型”，而不是全局均值
            unk_dist = torch.cdist(
                fg_features, self.unknown_centers.to(fg_features.device), p=self.margin
            ).min(dim=1).values
            # 用 cat 替换第 unknown 列，避免对 cdist 的输出做 in-place（会破坏 autograd）
            distances = torch.cat(
                [
                    distances[:, : self.unknown_idx],
                    unk_dist.unsqueeze(1),
                    distances[:, self.unknown_idx + 1 :],
                ],
                dim=1,
            )

        labels = []
        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                # 本类列 +1、其他列 -1。队列为空的类在距离矩阵里用零原型占位，
                # 标签仍记为 -1，保持与 ORE 原实现相同的 hinge 目标。
                if classes[index] == cls_index:
                    labels.append(1)
                else:
                    labels.append(-1)
        labels_t = torch.tensor(labels, device=fg_features.device).reshape(
            (-1, self.num_classes + 1)
        )

        if self.unk_memory_enable and self.unknown_centers is not None and unk_mask.any():
            storage = get_event_storage()
            loss_mat = self.hingeloss_none(distances, labels_t)
            row_w = torch.ones(loss_mat.shape[0], device=loss_mat.device)
            unk_feats = fg_features[unk_mask]
            assign = torch.cdist(
                unk_feats, self.unknown_centers.to(unk_feats.device), p=self.margin
            ).argmin(dim=1)
            counts = torch.bincount(
                assign, minlength=self.unknown_centers.shape[0]
            ).float()

            if self.unk_weight_mode == "inv_freq":
                cnt = counts[assign]
                w = cnt.clamp(min=1.0).pow(-self.unk_alpha)
                # 4.4 最小可靠簇下限：单样本/双样本小簇不把权重抬到 1.0，
                # 而是压到“最小可靠簇”的水平，避免噪声簇被过度强调。
                if self.unk_min_cluster > 0:
                    small = cnt < float(self.unk_min_cluster)
                    if small.any():
                        w = torch.where(
                            small,
                            torch.full_like(w, float(self.unk_min_cluster) ** (-self.unk_alpha)),
                            w,
                        )
            elif self.unk_weight_mode == "novelty":
                sim = F.cosine_similarity(
                    unk_feats[:, None, :],
                    self.unknown_centers.to(unk_feats.device)[None, :, :],
                    dim=2,
                ).max(dim=1).values
                w = 1.0 - sim
            else:
                w = torch.ones(assign.shape[0], device=assign.device)

            # warmup 阶段权重全为 1，避免早期子原型不稳定时放大噪声。
            if self.unk_warmup_iters > 0 and storage.iter < self.unk_warmup_iters:
                w = torch.ones_like(w)
            w_raw = w
            # 4.4 权重上限裁剪 + 均值归一化（只作用于 unknown 行）。
            if self.unk_weight_clip > 0:
                w = w.clamp(max=float(self.unk_weight_clip))
            if self.unk_normalize:
                w = w / w.mean().clamp(min=1e-6)
            row_w[unk_mask] = w
            loss = (loss_mat * row_w[:, None]).sum() / labels_t.numel()

            storage.put_scalar("unk_memory/weight_raw_max", float(w_raw.max().item()))
            storage.put_scalar("unk_memory/weight_mean", float(w.mean().item()))
            storage.put_scalar("unk_memory/weight_max", float(w.max().item()))
            storage.put_scalar(
                "unk_memory/unk_queue",
                # Do NOT call retrieve(-1) here: it loops over all 82 classes and
                # converts every 2048-d feature to a Python list on every iteration.
                # Access the unknown deque directly; this is the hot logging path.
                len(self.feature_store.store[self.unknown_idx]),
            )
            if counts.numel() > 0:
                storage.put_scalar("unk_memory/batch_min_cluster", float(counts.min().item()))
                storage.put_scalar("unk_memory/batch_max_cluster", float(counts.max().item()))
        else:
            loss = self.hingeloss(distances, labels_t)

        return loss

    def get_clustering_loss(self, input_features, proposals):
        if not self.enable_clustering:
            return 0

        storage = get_event_storage()

        # --resume 时 storage.iter 已越过 START_ITER，旧的初始化分支不会执行；
        # 这里检测到 means 还是空的，就从已保存的 feature_store 重建。
        if storage.iter > self.clustering_start_iter and all(m is None for m in self.means):
            self._init_means_from_store()

        c_loss = 0
        if storage.iter == self.clustering_start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            self._update_unknown_centers(kmeans=False)
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif storage.iter > self.clustering_start_iter:
            if storage.iter % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(self.num_classes + 1)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if(mean) is not None and new_means[i] is not None:
                        self.means[i] = self.clustering_momentum * mean + \
                                        (1 - self.clustering_momentum) * new_means[i]

                self._update_unknown_centers(kmeans=True)

            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return c_loss

    # def get_ae_loss(self, input_features):
    #     # storage = get_event_storage()
    #     # ae_loss = 0
    #     # if storage.iter < self.clustering_start_iter :
    #     features_hat = self.ae_model(input_features)
    #     ae_loss = F.mse_loss(features_hat, input_features)
    #     return ae_loss

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals, input_features=None):
        """
        ROI Heads 训练阶段总入口。把 forward() 得到的分类/回归预测 + 采样后的 proposals，
        换算成一组可反向传播的 loss。相比原版 Faster R-CNN，这里多了两件事：
          ① 通过 invalid_class_range 抑制"本 task 还没引入的类"参与 CE（Sec 4.3 output layer）
          ② 训练时额外计算 OReg 对比聚类 loss（论文 Eq 1，Sec 4.1）
        框回归本体是 Faster R-CNN 原版，无改动。

        Args:
            predictions (tuple): forward() 的返回值 (scores, proposal_deltas)
                scores          : (R, num_classes+1) = (512, 82)  分类 logits
                                  列 0-79 已知类 / 列 80 unknown / 列 81 background
                proposal_deltas : (R, 4*(num_classes+1)) = (512, 328)
                                  每个类别一套 box 增量；class_agnostic_bbox_reg=True 时是 (R,4)
            proposals (list[Instances]):
                每个 Instances 一张图，携带
                    proposal_boxes : (R, 4)  由 _sample_proposals 返回的 sampled_idxs 定位
                    gt_classes     : (R,)    就是 _sample_proposals 返回的 gt_classes_ss
                                              取值 0-79 / 80(unknown) / 81(bg)
                    gt_boxes       : (M, 4)  该图的 GT 框
            input_features (Tensor | None):
                ROIAlign 之后、分类/回归头之前的特征，shape (R, C=1024)。
                训练时传入用来算 OReg；推理时 None，整块聚类 loss 跳过。

        Returns:
            Dict[str, Tensor]:
                {"loss_cls": ..., "loss_box_reg": ..., "loss_clustering": ...}
                每一项都已按配置里的权重缩放过
        """
        # 把 forward 的 tuple 拆成两个张量：分类 logits + box 增量
        scores, proposal_deltas = predictions

        # FastRCNNOutputs 不是 nn.Module，只是一个"打包 + 算 loss"的普通类。
        # 它的 .losses() 内部就一句：
        #     {"loss_cls": softmax_cross_entropy_loss(), "loss_box_reg": box_reg_loss()}
        # 其中 softmax_cross_entropy_loss 才是未见类抑制真正落地的地方（245 行）：
        #     self.pred_class_logits[:, self.invalid_class_range] = -10e10
        #     F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")
        # 本函数做的事，只是把 invalid_class_range 这个"要屏蔽的列下标列表"传下去
        losses = FastRCNNOutputs(
            self.box2box_transform,  # GT 框 <-> 增量参数 的变换工具（原版就有的）
            scores,  # (R, num_classes+1) 分类预测
            proposal_deltas,  # (R, 4*(num_classes+1)) 回归预测
            proposals,  # 采样后的 proposal + 其 gt_classes target
            self.invalid_class_range,  # 本 task 未引入的类别下标，例：Task1 时 = list(range(20, 80)) = [20..79]
            self.box_reg_exclude_unknown,  # 是否把 unknown 从 box_reg 里排除
            self.smooth_l1_beta,  # smooth L1 的 β（一般 1.0/9.0），标准参数
            self.box_reg_loss_type,  # "smooth_l1" / "l1"，一般不动
        ).losses()
        # 到这里 losses = {"loss_cls": <scalar>, "loss_box_reg": <scalar>}

        # 只有训练路径会进这个分支：input_features 是 ROI Heads 一路带上来的 (R,1024) feature
        # 推理路径 input_features=None，整块跳过 → 网络推理不产生 clustering 相关开销
        if input_features is not None:
            # 早期实验里作者试过 autoencoder 重构分支，最终版没启用，注释掉的死代码
            # losses["loss_cluster_encoder"] = self.get_ae_loss(input_features)

            # OReg 对比聚类 loss（论文 Eq 1）：
            #   拿当前 batch 每个 proposal 的 feature 和它的 gt_classes，
            #   与 Store 里缓存的每类原型 μ_k 比较，同类拉近、异类推远
            #   （HingeEmbeddingLoss margin=2，距离 cdist p=10）
            losses["loss_clustering"] = self.get_clustering_loss(input_features, proposals)

        # 按配置文件里的 LOSS_WEIGHT 逐项缩放；没配则乘 1.0
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            predictions,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    # def clstr_loss(self, input_features, proposals):
    #     """
    #     Get the foreground input_features, generate distributions for the class,
    #     get probability of each feature from each distribution;
    #     Compute loss: if belonging to a class -> likelihood should be higher
    #                   else -> lower
    #     :param input_features:
    #     :param proposals:
    #     :return:
    #     """
    #     loss = 0
    #     gt_classes = torch.cat([p.gt_classes for p in proposals])
    #     mask = gt_classes != self.num_classes
    #     fg_features = input_features[mask]
    #     classes = gt_classes[mask]
    #     # fg_features = self.ae_model.encoder(fg_features)
    #
    #     # Distribution per class
    #     log_prob = [None for _ in range(self.num_classes + 1)]
    #     # https://github.com/pytorch/pytorch/issues/23780
    #     for cls_index, mu in enumerate(self.means):
    #         if mu is not None:
    #             dist = Normal(loc=mu.cuda(), scale=torch.ones_like(mu.cuda()))
    #             log_prob[cls_index] = dist.log_prob(fg_features).mean(dim=1)
    #             # log_prob[cls_index] = torch.distributions.multivariate_normal. \
    #             #     MultivariateNormal(mu.cuda(), torch.eye(len(mu)).cuda()).log_prob(fg_features)
    #                 # MultivariateNormal(mu, torch.eye(len(mu))).log_prob(fg_features.cpu())
    #             #                     MultivariateNormal(mu[:2], torch.eye(len(mu[:2]))).log_prob(fg_features[:,:2].cpu())
    #         else:
    #             log_prob[cls_index] = torch.zeros((len(fg_features))).cuda()
    #
    #     log_prob = torch.stack(log_prob).T # num_of_fg_proposals x num_of_classes
    #     for i, p in enumerate(log_prob):
    #         weight = torch.ones_like(p) * -1
    #         weight[classes[i]] = 1
    #         p = p * weight
    #         loss += p.mean()
    #     return loss

    # def clstr_loss_l2(self, input_features, proposals):
    #     """
    #     Get the foreground input_features, generate distributions for the class,
    #     get probability of each feature from each distribution;
    #     Compute loss: if belonging to a class -> likelihood should be higher
    #                   else -> lower
    #     :param input_features:
    #     :param proposals:
    #     :return:
    #     """
    #     loss = 0
    #     gt_classes = torch.cat([p.gt_classes for p in proposals])
    #     mask = gt_classes != self.num_classes
    #     fg_features = input_features[mask]
    #     classes = gt_classes[mask]
    #     fg_features = self.ae_model.encoder(fg_features)
    #
    #     for index, feature in enumerate(fg_features):
    #         for cls_index, mu in enumerate(self.means):
    #             if mu is not None and feature is not None:
    #                 mu = mu.cuda()
    #                 if  classes[index] ==  cls_index:
    #                     loss -= F.mse_loss(feature, mu)
    #                 else:
    #                     loss += F.mse_loss(feature, mu)
    #
    #     return loss