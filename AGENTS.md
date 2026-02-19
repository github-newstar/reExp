# Agent 工作说明（reExp，当前状态）

## 1. 当前项目定位
- 本项目已从模板迁移为 `BraTS2023(3D)` 分割训练工程，核心框架为 `Hydra + Trainer + MONAI + PyTorch`。
- 主要参考实现：`refs/swin_unetr_brats23_segmentation_3d.py`（数据组织、训练/验证流程、指标体系）。
- 依赖管理约定：优先使用 `uv`。

## 2. 关键入口与调用链
- 训练入口：`train.py`
  - 读取 `src/configs/*.yaml`
  - 构建 dataloader/transforms
  - 实例化 model/loss/metrics/optimizer/scheduler
  - 进入 `Trainer.train()`
- Trainer 核心：
  - `src/trainer/base_trainer.py`
  - `src/trainer/trainer.py`
  - 训练阶段直接 `model(**batch)`；验证阶段可选 `sliding_window_inference`

## 3. 键名契约（必须保持）
- Dataset 输出：`image`, `label`, `case_id`
- Model 输入：`image`（`forward(self, image, **batch)`）
- Model 输出：`{"logits": ...}`
- Loss 输入：至少 `logits`, `label`，并返回 `{"loss": ...}`
- `trainer.device_tensors` 当前为 `["image", "label"]`

## 4. 数据与变换现状
- 数据集实现：
  - 原始 NIfTI：`src/datasets/brats23.py::BraTS23Dataset`
  - 磁盘缓存向量：`src/datasets/brats23.py::BraTS23CachedVectorDataset`
- 变换配置：
  - 原始：`src/configs/transforms/instance_transforms/brats23.yaml`
  - 缓存：`src/configs/transforms/instance_transforms/brats23_cached.yaml`
- 当前推理/验证尺寸策略：
  - `CropForegroundd(k_divisible=[32,32,32])`
  - `SpatialPadd([96,96,96])`
  - `CenterSpatialCropd([96,96,96])`
  - 目的：保证 batch 内形状一致，避免 `stack` 报错。

## 5. 滑动窗口验证（当前已接入）
- 代码位置：`src/trainer/trainer.py::_predict_logits`
- 开关与参数（在主配置中）：
  - `trainer.use_sliding_window_inference: True`
  - `trainer.sw_roi_size: [96, 96, 96]`
  - `trainer.sw_batch_size: 1`
  - `trainer.sw_overlap: 0.5`
- 默认在 `brats23_swin_unetr*.yaml` 中开启。

## 6. 可用模型（Hydra `model=`）
- `swin_unetr_brats23`：`src/model/swin_unetr.py`
- `lmambanet`：`src/model/lmambanet.py`
  - 使用 `mamba_ssm`，无 fallback（缺失会直接报错）
- `lgmambanet`：`src/model/lgmambanet.py`
  - 使用 GTS-Mamba 三分支瓶颈（同样强依赖 `mamba_ssm`）
- `unet3d`：`src/model/unet3d.py`
  - MONAI 官方 `UNet(spatial_dims=3)` 包装

## 7. 训练配置基线
- 主配置：
  - 原始数据：`src/configs/brats23_swin_unetr.yaml`
  - 缓存数据：`src/configs/brats23_swin_unetr_cached.yaml`
- 优化器与损失：
  - `AdamW(lr=1e-4, weight_decay=1e-5)`
  - `DiceSegLoss(sigmoid=True)`
- 监控指标：
  - `monitor: "max val_MeanDice"`

## 8. 缓存脚本（HDD 优化版）
- 脚本：`tools/prepare_brats_cache.py`
- 当前策略：
  - 前景裁剪 + 最小尺寸约束 + `k_divisible` 对齐
  - image 默认存 `float16`
  - label 存 `uint8` 单通道（加载时转 3 通道）
- 典型命令：
```bash
uv run python "tools/prepare_brats_cache.py" \
  --data-dir "<BraTS_root_with_BraTS-GLI-*>" \
  --cache-dir "<cache_dir>" \
  --overwrite-vectors \
  --no-uncompressed-nii \
  --image-dtype float16 \
  --crop-margin 4 \
  --min-size 96 96 96 \
  --k-divisible 32
```
- 注意：`--data-dir` 必须直接包含 `BraTS-GLI-*` 子目录。

## 9. 常用运行命令
- 原始数据训练（Swin）：
```bash
python train.py -cn=brats23_swin_unetr
```
- 缓存数据训练（Swin）：
```bash
python train.py -cn=brats23_swin_unetr_cached
```
- 切换到 LGMambaNet：
```bash
python train.py -cn=brats23_swin_unetr_cached model=lgmambanet
```
- 不跑 test 分区：
```bash
python train.py -cn=brats23_swin_unetr_cached '~datasets.test'
```

## 10. 运行注意事项
- W&B 横轴 `step` 是全局训练步，不是 epoch。
- 服务器上的cached数据集在/cloud/cloud-ssd1/cached
