# Agent 工作说明（reExp）

## 1. 项目定位
- 本项目是 `PyTorch 模板` 与 `即将开展的医学分割实验` 的融合工程。
- `refs/swin_unetr_brats23_segmentation_3d.py` 是后续网络与训练逻辑的主要参考来源（重点参考数据加载和训练流程，网络结构可替换）。
- 目标是将 `BITS 2023` 任务平滑接入当前模板化训练框架（Hydra + Trainer/Inferencer）。

## 2. 参考优先级
1. `refs/swin_unetr_brats23_segmentation_3d.py`
   - 提供 MONAI 风格的数据组织、变换、训练/验证流程、滑窗推理、Dice 评估与最佳模型保存策略。
2. 模板主流程（必须遵守）
   - `train.py`
   - `src/datasets/data_utils.py`
   - `src/trainer/base_trainer.py`
   - `src/trainer/trainer.py`
   - `src/configs/*.yaml`

## 3. 模板调用链（必须理解）
### 3.1 训练入口
- `train.py`
  - Hydra 读取 `src/configs/baseline.yaml`
  - `get_dataloaders(config, device)` 构建数据与 batch transforms
  - `instantiate(config.model/loss/metrics/optimizer/lr_scheduler)`
  - 创建 `Trainer(...).train()`

### 3.2 Trainer 核心流程
- `BaseTrainer._train_process()`
  - 每个 epoch 调 `self._train_epoch()`
  - 评估非 train 分区（val/test）
  - 按 `monitor` 指标判断 best / early stop
  - 保存 checkpoint（`model_best.pth`）

- `Trainer.process_batch()`
  - `move_batch_to_device` -> `transform_batch`
  - `outputs = model(**batch)`，并更新到 `batch`
  - `all_losses = criterion(**batch)`，并更新到 `batch`
  - 训练阶段执行反传、梯度裁剪、优化器与调度器 step
  - 更新 loss 与 metric 统计

### 3.3 推理入口
- `inference.py` -> `Inferencer.run_inference()`
- 可基于 `from_pretrained` 加载模型，并按分区保存预测。

## 4. 当前模板“键名契约”（非常重要）
以下字段名必须在 Dataset/Collate/Model/Loss/Metric 间一致：
- Dataset (`BaseDataset.__getitem__`) 默认返回：
  - `data_object`
  - `labels`
- Collate (`src/datasets/collate.py`) 组装同名 batch 字段。
- Model (`src/model/baseline_model.py`) 期望输入 `data_object`，输出：
  - `logits`
- Loss (`src/loss/example.py`) 期望输入：
  - `logits`, `labels`
  - 且必须返回包含 `loss` 的 dict。
- Metric (`src/metrics/example.py`) 期望输入：
  - `logits`, `labels`

如果改为医学分割常见键名（如 `image`/`label`），必须同步修改：
- dataset 返回字段
- collate_fn
- model forward 参数
- loss/metric 参数
- `trainer.device_tensors` 配置
- transforms 配置键名

## 5. REF 中可直接迁移的关键设计
基于 `refs/swin_unetr_brats23_segmentation_3d.py`：
- 数据清单构建：
  - 扫描病例目录，按病例组装多模态影像路径与标签路径。
  - 使用固定随机种子 + 比例采样 + K 折划分 train/val。
- 标签转换：
  - 自定义 `MapTransform` 将标签映射为多通道训练目标（TC/WT/ET 思路可迁移到 BITS 任务定义）。
- 数据变换：
  - 训练：Load -> 标签映射 -> 前景裁剪 -> 随机空间裁剪 -> 翻转/强度增强 -> 归一化。
  - 验证：Load -> 标签映射 -> 归一化（保持确定性）。
- 训练流程：
  - DiceLoss + AdamW + CosineAnnealingLR
  - 支持 AMP (`autocast` + `GradScaler`)
  - 定期验证，滑窗推理 `sliding_window_inference`
  - 计算多类别 Dice，按平均 Dice 保存最优模型。

## 6. BITS 接入模板的最小落地步骤（建议顺序）
1. 新建 `src/datasets/bits2023.py`
   - 生成 index（每条样本包含多模态路径、标签路径、病例 ID、fold 信息）。
   - 复用 `BaseDataset` 思路，但按 3D 医学影像读取逻辑实现。
2. 新建/调整 transforms
   - 在 `src/configs/transforms/` 新建面向 BITS 的 instance/batch transforms 配置。
   - 如使用 MONAI transforms，优先放 instance transforms；需要 GPU 上 batch transform 时放 batch transforms。
3. 更新 `collate_fn`
   - 支持体数据 batch（通常是 `[B, C, H, W, D]`）及元信息。
4. 新建模型封装
   - 例如 `src/model/swin_unetr_model.py`，forward 返回 `{"logits": ...}` 以兼容 Trainer。
5. 新建分割 loss/metrics
   - 例如 Dice 系列，返回格式严格遵守 `{"loss": ...}`。
6. 新建实验配置
   - `src/configs/model/swin_unetr.yaml`
   - `src/configs/datasets/bits2023.yaml`
   - `src/configs/metrics/segmentation.yaml`
   - `src/configs/bits_swin_unetr.yaml` 作为主入口。

## 7. 运行方式（模板约定）
- 训练：
  - `python3 train.py -cn=bits_swin_unetr`
- 推理：
  - `python3 inference.py model=... datasets=... inferencer.from_pretrained=...`

## 8. 实施约束
- 保持 KISS：先跑通单卡、单 fold、固定数据比例的小闭环，再扩展。
- 保持 YAGNI：当前只实现“可训练 + 可验证 + 可保存 best”最小能力。
- 保持 DRY：路径扫描、fold 划分、Dice 计算封装成复用函数，避免脚本散落复制。
- 保持一致性：字段命名与 Hydra 配置键严格对齐，避免隐式耦合。

## 9. 下一阶段建议（执行优先级）
1. 先确定 BITS 数据目录结构与标签定义映射（是否多类别/多通道）。
2. 先完成 Dataset + Config + Dummy Model 的端到端冒烟训练。
3. 再接入 Swin UNETR 与滑窗验证，最后补全可视化与推理导出。
