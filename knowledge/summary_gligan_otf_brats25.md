# Paper Summary: On-the-Fly Data Augmentation for Brain Tumor Segmentation (arXiv:2509.24973)

- Paper: https://arxiv.org/abs/2509.24973
- TeX source: https://arxiv.org/src/2509.24973

## 1) 论文方法核心

这篇工作不是通用“自动增强策略搜索”，而是面向 BraTS 的工程化方案：

1. 用预训练 GliGAN 在训练时动态生成“插入式”合成肿瘤（on-the-fly），替代离线大规模预生成。
2. 在每个训练 batch 前，以概率 `p` 决定是否对输入做合成插瘤；验证集不做增强。
3. 合成时使用“标签修改器（label modifier）”控制插入肿瘤形态与类别分布：
   - 类别改写（针对类别不平衡）：以一定概率把 SNFH/ET 改写为更稀缺类别。
   - 尺度缩放：按范围缩小病灶，提升小病灶样本频率。
   - 多病灶插入：支持最多插入 2 个合成病灶（额外概率触发第二次插入）。
4. 最终采用 baseline + on-the-fly 增强模型做集成，提升鲁棒性。

一句话概括：**把“合成病灶插入”变成训练时在线操作，并通过标签级控制强化稀缺/小病灶学习。**

## 2) 可迁移到你当前项目的部分

你的仓库是 `Hydra + MONAI + 自研 Trainer`，不是 nnU-Net；可直接迁移的是“策略思想”，不是其 nnU-Net 代码。

可迁移点：

1. 训练时在线插瘤（而非离线缓存增强样本）。
2. 按概率触发插瘤（混合原始样本与合成样本）。
3. 标签修改策略：类别重分布 + 小病灶尺度扰动 + 多病灶数量控制。
4. 增强模型与非增强模型的后期集成。

需重实现点：

1. GliGAN 生成器推理接口（输入/输出与你当前 BraTS 张量格式对齐）。
2. “健康区域插入”逻辑（插入位置约束、重叠规则、边界平滑）。
3. 训练流水线中的 on-the-fly transform（MONAI MapTransform 风格）。

## 3) 在本实验中的实现计划（建议三阶段）

### Phase A: 最小可用版本（先跑通）

目标：先把 on-the-fly 插瘤接进训练，不引入复杂策略。

1. 新增模块 `src/transforms/gli_otf.py`
   - `RandGliGanInsertd(keys=["image","label"], prob, max_insertions, scale_range, class_rewrite_prob, ...)`
   - 输入输出保持你项目契约：`image/label/case_id`。
2. 在 `src/configs/transforms/instance_transforms/` 新增配置
   - `brats23_cached_gli_otf.yaml`。
   - 放置位置：建议在 `RandETFocusedCropd` 之后、常规强度增强之前。
3. 先用“占位生成器”验证流程（例如从库中采样 label patch + 强度混合），确认训练稳定。
4. 记录日志
   - 每轮统计：触发率、平均插入个数、各类别插入体素比例。

交付标准：
- 训练可稳定跑完 `ultrafast`。
- 无 shape/键名回归。
- 可见增强触发统计。

### Phase B: 论文策略对齐版本（核心）

目标：实现论文里的“标签修改器”逻辑。

1. 接入真实 GliGAN 推理（预训练权重 + modality 对齐）。
2. 实现 label modifier：
   - 类别改写（例如 `SNFH->ET`, `ET->NETC` 概率链）。
   - 尺度策略（含“是否去掉 SNFH”分支下不同 scale 范围）。
   - 第二病灶触发概率与最多插入数。
3. 插入约束
   - 只在健康区插入；与已有病灶重叠阈值限制；越界裁剪。
4. 新增 Hydra 参数
   - `trainer.gli_otf.*` 或 `transforms.instance_transforms.train.transforms[*].*`

交付标准：
- 可复现两种策略：Regular / Custom（对应文中 model2/model3 风格）。
- 训练吞吐下降可控（建议 <20% 作为初始目标，后续再优化）。

### Phase C: 实验与集成验证

目标：验证是否对 BraTS 指标有效。

1. A/B 方案
   - Baseline（现有增强）
   - Baseline + Regular OTF
   - Baseline + Custom OTF
2. 统一训练配置（epoch、batch、seed）做公平对比。
3. 指标
   - MeanDice + 各子区 Dice + HD95。
   - ET/TC 小病灶召回（重点观察）。
4. 若单模型提升有限，尝试 checkpoint 集成（你现有工具链可支持）。

## 4) 与当前工程的具体改动清单

1. 新文件
   - `src/transforms/gli_otf.py`
   - （可选）`src/augmentation/gli_generator.py`
   - `src/configs/transforms/instance_transforms/brats23_cached_gli_otf.yaml`
2. 修改
   - `src/transforms/__init__.py` 导出新 transform。
   - `src/configs/transforms/brats23_cached_gli_otf.yaml`（组合 batch/instance transform）。
   - 新建实验 config（如 `*_gli_otf_cached_ep100_ultrafast.yaml`）。
3. 不改动
   - 数据键名契约、model 输入输出契约、loss 接口契约。

## 5) 主要风险与规避

1. 生成质量不足导致“伪病灶噪声”
   - 先低触发概率（如 `p=0.2~0.4`）+ 小范围 scale。
2. 训练耗时上升
   - 预热缓存 GAN 到 GPU；控制每 batch 最大插入次数。
3. 类别映射错误（BraTS2025 四类 vs 你当前三通道标签）
   - 先明确映射策略（例如在 3 通道训练标签中如何承载 RC/NETC/SNFH 信息）。
4. 与现有随机裁剪冲突
   - 建议先 crop 后插瘤，减少无效生成；再做强度增强。

## 6) 关键现实约束（重要）

论文依赖预训练 GliGAN 权重和其生成逻辑。若当前环境没有可用权重/推理代码，本计划可先落地“接口+占位生成器”，再替换为真实 GliGAN。

