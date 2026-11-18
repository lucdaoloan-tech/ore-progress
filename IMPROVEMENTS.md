# OWOD 伪未知监督改进：代码实现说明（v2）

本文件对应《OWOD_伪未知监督改进方法说明》v2，记录代码入口、默认配置与验证方式。
所有改动只发生在训练侧；检测器结构、推理图与可学习参数数量不变。

## 1. 伪未知选样（文档 §3）

入口：`detectron2/modeling/roi_heads/roi_heads.py`

- `_sample_proposals` 额外接收 `proposal_boxes` 与 `gt_boxes`，在 ROI 采样之外仍可访问全部 RPN proposal。
- `_select_unknown_candidates` 按顺序执行：
  1. 候选池 `matched_label == 0`（`UNK_SELECT.ALL_PROPOSALS=False` 时退化为只在 ROI 采样池内选）；
  2. 已知 GT 抑制：候选与任一已知 GT 的最大 IoU `> GT_IOU_MAX` 时剔除（默认 0.3，0 关闭）；
  3. 每图 objectness 分位数 `max(Quantile_q, MIN_SCORE)`（默认 q=0.95、s_min=0.2）；
  4. 类无关贪心 NMS（IoU 0.5）与每图上限 m=3；
  5. 候选数 `< MIN_CANDIDATES=20` 时改用 top-m 兜底，不计算分位数；
  6. 选中的 proposal 标为 unknown（索引 80），未进入 ROI 采样预算的候选追加进当前图像训练批次。
- Task 2+：未来类 GT（`cls >= PREV+CUR`）的匹配标签被改回背景，避免未来类物体被当成 invalid
  foreground 而失去“伪未知候选”资格；GT 抑制只使用 `PREV+CUR` 的已知类 GT。
- `_topk_background` 保留 ORE 原始 top-k 行为，供 `UNK_SELECT.ENABLE=False` 时复现 baseline。
- 诊断标量：`unk_select/num_selected`、`num_bg`、`thr`、`zero_ratio`、`num_suppressed_gt`。

## 2. 未知记忆与 OReg 加权（文档 §4）

入口：`detectron2/modeling/roi_heads/fast_rcnn.py`

- `_update_unknown_centers`：
  - `CLUSTERING.START_ITER` 时用 unknown 队列均值初始化单原型（不提前 k-means）；
  - 之后每个 `UPDATE_MU_ITER` 用最远点初始化 + 10 轮 k-means 得到 K=4 个子原型，丢弃空簇；
  - 新旧中心做 EMA（ρ=`CLUSTERING.MOMENTUM`，默认 0.99）。
- `clstr_loss_l2_cdist`：
  - unknown 列距离改为 `min_k ||f_i - μ_k||_p`（p=10），其余列仍为单原型距离；
  - unknown 行权重 `w_i = n_{a(i)}^{-α}`；`n < MIN_CLUSTER` 时压到 `MIN_CLUSTER^{-α}`；再按 `WEIGHT_CLIP` 截断、按均值归一化（`NORMALIZE=True`）；
  - warmup（默认前 3000 iter）权重全 1；已知类行权重 1；标签矩阵与 hinge 形式不变。
- 诊断标量：`unk_memory/num_centers`、`queue_min_cluster`、`queue_max_cluster`、`batch_min_cluster`、`batch_max_cluster`、`weight_raw_max`、`weight_max`、`weight_mean`、`unk_queue`。

## 3. 框回归与推理一致性

- `BOX_REG_EXCLUDE_UNKNOWN=True`：`FastRCNNOutputs.box_reg_loss` 排除 unknown 类（索引 80），避免伪未知噪声框参与回归。
- `PascalVOCDetectionEvaluator.update_label_based_on_energy` 改为与 Weibull 拟合一致的
  `E = T·logsumexp(z/T)`（T=`OWOD.TEMPERATURE`，默认 1.5），修复原先评估侧未乘温度的不一致。
- `train_loop.analyse_energy` 的已知/未知划分改为 `cls >= num_seen_classes`（背景除外），
  与评估器 `parse_rec` 把“所有未引入类”映射为 unknown 的协议一致；原实现只把索引 80 当 unknown，
  会把 20–79 的 COCO 真实未知类错误计入 known 能量。
- 兼容性修复：`pascal_voc.py` 等 VOC 加载器把 NumPy 已移除的 `np.str` 改为 `str`（numpy ≥ 1.24 必需）。

## 3.1 空原型与标签语义

- 未知/已知类队列为空时保持原型为 `None`（不再原地写零）；距离矩阵中用零向量占位，
  标签仍记 `-1`。这样既符合文档“原型保持为空，直到首次出现样本”，又与 ORE 原实现的 hinge 目标数值一致。
  （注意：`HingeEmbeddingLoss` 在 target=0 时按正样本处理，不能把空类标签写成 0。）

## 4. 配置

- `configs/OWOD/t1/t1_train_unk.yaml`：方法配置（选样 + 记忆 + GT 抑制 + 噪声控制）。
- `configs/OWOD/t1/t1_train_base_clean.yaml`：同代码基线的 ORE-style 对照（选样/记忆关闭，box-reg 处理相同）。
- `configs/OWOD/t1/t1_train_unk_short.yaml`：2000 iter 快速筛查配置。
- `configs/OWOD/t2|t3|t4/*_train_unk.yaml`：后续任务沿用同一机制。
- 新增配置项：`OWOD.UNK_SELECT.GT_IOU_MAX`、`OWOD.UNK_MEMORY.MIN_CLUSTER`、`WEIGHT_CLIP`、`NORMALIZE`。

## 5. 诊断工具

- `tools/diagnose_unknown_queue.py`：队列长度、特征范数、两两余弦、K=4 簇大小/熵、可选 energy AUROC（温度与评估一致）。
- `tools/diagnose_unknown_predictions.py`：预测 unknown 与原始 XML 匹配，输出 unknown 纯度、真实类别分布/熵、逐类覆盖；使用与测试相同的 energy 重标定。

## 6. 验证

- 单元测试：`tests/test_owod_improvements.py`（GT 抑制、分位数/top-k、NMS、采样追加、子原型、加权、box-reg 排除）。
- 真实数据冒烟：`output/smoke_v2`（30 iter，2×4090），日志中可见 `unk_select/num_selected=3`、`num_suppressed_gt≈80–90`、`unk_memory/num_centers` 从 1 增至 2–4。
- 正式训练：`configs/OWOD/t1/t1_train_unk.yaml`，2 卡、全局 batch 8、LR 0.01、18,000 iter。
  截至 13,459 iter：`num_selected=3` 恒定、`zero_ratio=0`、每图被 GT 抑制 81–201 个候选、
  `num_centers` 1→4、`weight_raw_max` 在 warmup 后为 0.707（最小簇下限）、归一化后 `weight_max≈1.45–1.66`（≤ clip 2.0）、
  `loss_clustering≈0.0052`；训练损失从 2.15 降到 0.45。

## 7. 2026-11-13 code-review follow-up

- **性能回退修复**：`unk_memory/unk_queue` 日志不再每轮调用 `feature_store.retrieve(-1)`。
  原写法会遍历 82 个类并把每个 2048 维特征转成 Python list；现改为直接读
  `len(self.feature_store.store[self.unknown_idx])`。`retrieve(-1)` 仅保留在每 3000 iter
  的聚类/初始化路径。
- **`future_match` 防御性 guard**：对 `matched_idxs >= 0` 的 proposal 才做未来类判断。
  当前 Detectron2 `Matcher` 返回的是 `max` 的 GT index，正常不会出现 `-1`；guard 不影响
  Task1，但可避免其他 matcher 实现下 `-1` 索引到最后一个 GT。
- **`NORMALIZE` 一致性**：代码与说明文档均保留 `UNK_MEMORY.NORMALIZE=True` 和均值归一化；
  若论文正文写“删除 NORMALIZE”，需要改论文文字，而不是改代码。

