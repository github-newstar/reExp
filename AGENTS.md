# Agent 工作说明（reExp，当前状态）

## 1. 项目定位
- 任务：`BraTS2023` 3D 脑肿瘤分割。
- 框架：`Hydra + PyTorch + MONAI + 自研 Trainer`。
- 训练入口：`train.py`。
- 配置根目录：`src/configs/`。

## 2. 关键约定（必须保持）
- Dataset 输出键：`image`, `label`, `case_id`。
- Model 输入：`forward(self, image, **batch)`。
- Model 输出：`{"logits": ...}`。
- Loss 输入：至少包含 `logits`, `label`，并返回 `{"loss": ...}`。
- `trainer.device_tensors`：`["image", "label"]`。

## 3. 当前主网络（LGMamba LightFSDE）
文件：`src/model/lgmamba_fsde.py`

- 主干：`LGMambaLightFSDENet`
  - 编码器：3 层（`enc1/2/3` + `down1/2/3`）
  - 瓶颈：`GTSMambaBottleneck`（三分支 tri-axis Mamba）
  - 解码器：3 层（`up3/2/1` + `dec3/2/1`）
  - 跳跃增强：`skip_fsde1/2/3`（`LightFSDEBlock`）
  - 输出头：`head`，训练态可输出深监督 `aux_logits`

- `LightFSDEBlock` 特点
  - 空间分支：`DWConv3d + IN + SiLU`
  - 频域分支：`rfftn/irfftn` + 可学习小频谱权重（上采样到当前频域尺寸）
  - 门控：`1x1x1 Conv + Sigmoid`

- 深监督行为
  - 仅训练态输出 `aux_logits`；验证/推理态不输出，减少显存和计算。

## 4. ECA 模块与开关
文件：`src/model/lmambanet.py`

- 核心模块：`ECABlock3D`
  - `GAP -> 1D Conv(自适应核长) -> Sigmoid`，做通道重标定。
- `DIDCBlock` 内部的 `channel_interaction` 默认用 ECA；若 `use_channel_shuffle=False` 则退化为 `Identity`。
- 在 LGMamba 系列里，ECA 主要出现在：
  - 编解码 `DIDCBlock` 的 `channel_interaction`
  - 瓶颈内部（不同变体前置/后置/双 ECA/无 ECA）

## 5. 瓶颈层清单（GTSMamba 系列）
文件：`src/model/lgmambanet.py`

- `GTSMambaBottleneck`：默认版，tri-axis Mamba + post-ECA
- `GTSMambaBottleneckPreECA`：Pre-ECA
- `GTSMambaBottleneckNoECA`：无 ECA
- `GTSMambaBottleneckPrePostECA`：Pre+Post ECA
- `GTSMambaBottleneckSpatialPrior`：Pre DWConv3x3x3 + LayerNorm + tri-axis Mamba
- `GTSMambaBottleneckResidualInject`：旁路残差注入（1x1+Sigmoid 门控）
- `GTSMambaBottleneckECAMambaECAMamba`：`ECA -> Mamba -> ECA -> Mamba`
- `GTSMambaBottleneckMambaECAMambaECA`：`Mamba -> ECA -> Mamba -> ECA`（第二套独立 Mamba 参数）
- `GTSMambaBottleneckDWConvMambaDWConvMamba`：`DW -> Mamba -> DW -> Mamba`
- `GTSMambaBottleneckDWConvMambaECAMambaECA`：`DW -> Mamba -> ECA -> Mamba -> ECA`
- `GTSMambaBottleneckDWConvECAMambaECAMamba`：`DW -> ECA -> Mamba -> ECA -> Mamba`

## 6. 可用模型（Hydra `model=`）
目录：`src/configs/model/`

- 运行时导出全集（`src/model/__init__.py`）
  - `BaselineModel`
  - `LGMambaFSDENet`
  - `LGMambaLightFSDENet`
  - `LGMambaLightFSDENoShuffleNet`
  - `LGMambaLightFSDEStage123NoECAPostECANet`
  - `LGMambaLightFSDEBottleneckNoECANet`
  - `LGMambaLightFSDEPreECANet`
  - `LGMambaLightFSDEPrePostECANet`
  - `LGMambaLightFSDESpatialPriorNet`
  - `LGMambaLightFSDEResidualInjectNet`
  - `LGMambaLightFSDEShallowPlainNet`
  - `LGMambaLightFSDEShallowSkip12NoECANet`
  - `LGMambaLightFSDEShallowECAMambaECAMambaNet`
  - `LGMambaLightFSDEShallowSkip12NoECAECAMambaECAMambaNet`
  - `LGMambaLightFSDEShallowSkip12NoECAECAMambaECAMambaDec3NoECANet`
  - `LGMambaLightFSDEShallowSkip12NoECA_MambaECAMambaECANet`
  - `LGMambaLightFSDEShallowSkip12NoECA_DWConvMambaECAMambaECANet`
  - `LGMambaLightFSDEShallowSkip12NoECA_DWConvECAMambaECAMambaNet`
  - `LGMambaLightFSDEShallowSkip12NoECA_DWConvMambaDWConvMambaNet`
  - `LGMambaNet`
  - `LMambaNet`
  - `UNet3D`
  - `SwinUNETRSegModel`
  - `NoNewNet`

- 基础模型
  - `lgmamba_lightfsde` -> `LGMambaLightFSDENet`
  - `lgmamba_fsde` -> `LGMambaFSDENet`（向后兼容别名）
  - `lgmambanet` -> `LGMambaNet`
  - `lmambanet` -> `LMambaNet`
  - `unet3d` -> `UNet3D`
  - `swin_unetr_brats23` -> `SwinUNETRSegModel`
  - `no_new_net` -> `NoNewNet`

- LGMamba LightFSDE 变体
  - `lgmamba_lightfsde_noshuffle`
  - `lgmamba_lightfsde_pre_eca`
  - `lgmamba_lightfsde_bottleneck_no_eca`
  - `lgmamba_lightfsde_pre_post_eca`
  - `lgmamba_lightfsde_spatial_prior`
  - `lgmamba_lightfsde_residual_inject`
  - `lgmamba_lightfsde_stage123_no_eca_post_eca`
  - `lgmamba_lightfsde_shallow_plain`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_eca_mamba_eca_mamba`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_eca_mamba_eca_mamba_dec3_no_eca`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_mamba_eca_mamba_eca`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_mamba_eca_mamba_eca`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_eca_mamba_eca_mamba`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_mamba_dwconv_mamba`

## 7. 训练配置（重点）

### 7.1 300 epoch 基线
文件：`src/configs/lgmamba_lightfsde_cached_ep300_policy.yaml`

- `batch_size=6`
- `n_epochs=300`
- warmup：`30` epoch
- `max_grad_norm=3.0`
- quick 验证策略：
  - 前 50%：每 5 轮
  - 50%-80%：每 3 轮
  - 后 20%：每 2 轮
- 训练后 full-eval：
  - `post_training_full_eval.enabled=true`
  - `top_k=5`
  - 默认会输出排序日志，并写 `post_full_eval_summary.json`

### 7.2 300 epoch 变体策略
- `lgmamba_lightfsde_pre_eca_cached_ep300_policy`
- `lgmamba_lightfsde_pre_post_eca_cached_ep300_policy`
- `lgmamba_lightfsde_bottleneck_no_eca_cached_ep300_policy`
- `lgmamba_lightfsde_spatial_prior_cached_ep300_policy`
- `lgmamba_lightfsde_residual_inject_cached_ep300_policy`

### 7.3 UltraFast（100 epoch）配置
目录：`src/configs/*ultrafast*.yaml`

- 共性（多数配置）
  - `n_epochs=100`
  - warmup：`10` epoch
  - `batch_size=2`
  - quick 验证：前 50% 每 3 轮，后 50% 每 2 轮
  - `full_eval.epochs=[100]`（最后一轮 full）
  - `post_training_full_eval.top_k=3`
  - `report_best_only=true`

- 当前已存在的 ultrafast 配置
  - `lgmamba_lightfsde_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_pre_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_post_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_bottleneck_no_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_residual_inject_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_plain_post_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_stage123_no_eca_post_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_eca_mamba_eca_mamba_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_eca_mamba_eca_mamba_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_eca_mamba_eca_mamba_dec3_no_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_mamba_eca_mamba_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_mamba_eca_mamba_eca_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_eca_mamba_eca_mamba_cached_ep100_ultrafast`
  - `lgmamba_lightfsde_shallow_skip12_no_eca_dwconv_mamba_dwconv_mamba_cached_ep100_ultrafast`
  - `unet3d_cached_ep100_ultrafast`

- 注意
  - `lgmamba_lightfsde_cached_ep100_ultrafast.yaml` 当前仍是全程 `interval=2`（`q2`），与其余 `3/2` 策略不同。

## 8. 自动续训与验证机制

### 8.1 自动续训
文件：`train.py`

- 默认 `trainer.auto_resume=true`（若配置未显式关闭）。
- 续训查找顺序：
  1. `model_best.pth`
  2. 最新 `checkpoint-epoch*.pth`
- 命中后自动设置：
  - `trainer.resume_from=<ckpt>`
  - `trainer.override=false`

### 8.2 checkpoint 兼容加载
文件：`src/trainer/base_trainer.py`

- 通过 `_load_checkpoint_compat` 处理 PyTorch 2.6 `weights_only` 变化，避免历史 checkpoint 反序列化问题。

### 8.3 训练后 top-k 全量验证
文件：`src/trainer/base_trainer.py::_run_post_training_full_eval`

- 从 quick 评估历史筛选 top-k epoch
- 对候选 checkpoint 做 full-eval
- 记录排名、最佳项，并保存 summary JSON
- 当 `report_best_only=true` 仅打印最佳候选

## 9. 数据与缓存
- 原始数据集：`src/datasets/brats23.py::BraTS23Dataset`
- 缓存数据集：`src/datasets/brats23.py::BraTS23CachedVectorDataset`
- 缓存预处理脚本：`tools/prepare_brats_cache.py`
- 服务器常用缓存目录：`/cloud/cloud-ssd1/cached`

## 10. 常用评估/分析脚本

- 按 run_name 自动找 best ckpt 做 full val：
  - `tools/run_full_val_from_run_name.py`
  - 优先找 `best_model.pth` / `model_best.pth`，否则取最新 epoch checkpoint

- 模型 profile（参数量/时延/FLOPs）：
  - `tools/profile_model_server.py`
  - 支持两种入口：`--config-name` 或 `--run-name`（二选一）
  - `--run-name` 模式会自动读取 `saved/<run_name>/config.yaml` 并自动选 best checkpoint
  - `--no-torch-profiler` 用于与 `ncu` 同跑时避免 CUPTI 冲突

- NCU 结果 FLOPs 汇总：
  - `tools/summarize_ncu_flops.py`

## 11. 运行命令示例
- 300 epoch 基线：
```bash
python train.py -cn=lgmamba_lightfsde_cached_ep300_policy
```
- 300 epoch Pre-ECA：
```bash
python train.py -cn=lgmamba_lightfsde_pre_eca_cached_ep300_policy
```
- 100 epoch UltraFast（默认 post-ECA 主干）：
```bash
python train.py -cn=lgmamba_lightfsde_post_eca_cached_ep100_ultrafast
```
- 100 epoch UltraFast（1,2 层无 ECA + `Mamba->ECA->Mamba->ECA`）：
```bash
python train.py -cn=lgmamba_lightfsde_shallow_skip12_no_eca_mamba_eca_mamba_eca_cached_ep100_ultrafast
```

## 12. 依赖与兼容性备注
- `mamba_ssm` 为 L/Mamba 相关模型硬依赖。
- `train.py` 启动时会：
  - `patch_monai_numpy_dtype_compat()`
  - `set_track_meta(False)`（兼容不同 MONAI 版本导入路径）
- `pyproject.toml` 已包含 `fvcore`，用于 graph-based FLOPs 统计。
