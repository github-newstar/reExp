# Paper Summary: SADA (On-the-Fly Data Augmentation via Gradient-Guided and Sample-Aware Influence Estimation)

- Paper: https://arxiv.org/html/2510.00434v1
- Source bundle: https://arxiv.org/src/2510.00434v1

## 1. 核心设计思路（论文方法）

SADA 关注的问题不是“选什么增强操作”，而是“每个样本此刻该增强多强”。

### 1.1 关键假设

- 样本难度随训练阶段动态变化。
- 静态/随机统一强度增强会与当前训练需求错配。

### 1.2 核心机制

1. **样本影响估计**
   - 目标是估计样本对当前优化方向的贡献稳定性。
   - 理想形式：样本梯度与参数更新方向的投影。
   - 实用近似：用相邻时刻输出/损失差分近似（文中给出 KL 形式）。

2. **稳定性量化**
   - 在长度为 `L` 的局部窗口上计算影响量的方差 `V_t(x_i)`。
   - 再用 EMA（系数 `beta`）平滑得到 `V(x_i)`。

3. **强度映射**
   - 对 `V(x_i)` 做 min-max 归一化，并反向映射到 `s(x_i) in [0,1]`：
   - 方差小（稳定样本）=> `s` 大 => 更强增强；
   - 方差大（不稳定样本）=> `s` 小 => 更弱增强。

4. **增强执行策略**
   - 每个样本每个 epoch 随机抽一个增强操作；
   - 用 `s(x_i) * m_max(op)` 调整该操作幅度。

### 1.3 经验结论

- 小窗口 `L` 更好（响应更及时）。
- `beta` 相对鲁棒。
- 不依赖额外策略网络/双层优化，开销小。

---

## 2. 面向 reExp (BraTS23 3D 分割) 的落地方案

当前工程特征：
- 训练数据增强在 `instance_transforms`（MONAI）中完成；
- batch 含 `case_id`，可用于样本级状态跟踪；
- 训练入口 `train.py`，核心循环在 `src/trainer/trainer.py`。

### 2.1 迁移原则

- 论文是 2D 分类，reExp 是 3D 分割。
- 迁移时保留“**样本感知动态强度**”思想，不照搬 2D 操作空间。

建议定义 3D 操作空间（仅可控强度项）：
- Spatial: rotate/scale (RandAffined), flip（概率可映射）
- Intensity: noise std, smooth sigma, scale factor, shift offset, contrast gamma

### 2.2 推荐的 SADA-v1（低风险）实现

> 先做“**按 case 动态强度** + **每 epoch 更新一次强度**”，避免多进程 worker 共享状态复杂度。

#### A. 统计器（Trainer 侧）

新增 `SADAController`：
- 以 `case_id` 为 key 保存：
  - `prev_loss`（或 proxy）
  - `delta_history`（deque, 长度 L）
  - `ema_var`
  - `strength`
- 每个训练 batch 后更新：
  - 计算每样本 `delta = abs(loss_i_t - prev_loss_i)`
  - 更新局部方差 + EMA

> v1 推荐直接用每样本分割损失（CE/Dice 组合可拆到 sample 级），而不是存储前一时刻全量 logits，显存和实现更稳。

#### B. 强度发布（Epoch 边界）

- 每个 epoch 结束后，在主进程:
  - 全体样本 `ema_var` 做 min-max 归一化
  - 计算 `s = 1 - norm(var)`
  - 写入 `saved/<run_name>/sada_strength_epoch_{k}.json`

#### C. 增强执行（Transform 侧）

新增 `src/transforms/sada.py`：
- `SADADynamicAugmentd`
  - 输入样本 dict（包含 `image`,`label`,`case_id`）
  - 查当前 epoch 对应 `case_id` 的 `s`
  - 随机选 1 个增强 op
  - 按 `m = s * m_max(op)` 应用

放置位置：
- 在 `RandETFocusedCropd` 后、`EnsureTyped` 前
- 替换/融合现有固定幅度强增强（避免重复过强）

#### D. 配置扩展

在训练 config 增加：
- `trainer.sada.enabled`
- `trainer.sada.window_size` (`L`)
- `trainer.sada.ema_beta`
- `trainer.sada.warmup_epochs`（前几轮先用固定强度）
- `trainer.sada.update_interval`（v1 用 1 epoch）
- `trainer.sada.op_space`
- `trainer.sada.strength_file`

### 2.3 v2（更接近论文“on-the-fly”）

- 从 epoch 级更新升级为 step 级更新。
- 需要 worker 可见共享状态（`multiprocessing.Manager().dict()` 或主进程 batch-transform 实现）。
- 复杂度高，建议在 v1 验证收益后再做。

---

## 3. 在本项目的具体改动点（建议）

1. 新增 `src/augmentation/sada_controller.py`
2. 新增 `src/transforms/sada.py`
3. 修改 `src/transforms/__init__.py` 导出新 transform
4. 修改 `src/trainer/trainer.py`
   - 在训练 batch 后调用 controller.update(batch)
   - 在 epoch 结束时 controller.publish_strengths()
5. 新增 `src/configs/transforms/instance_transforms/brats23_cached_sada.yaml`
6. 新增实验配置（例如）
   - `src/configs/liunet_mkir_drbd_sada_cached_ep100.yaml`

---

## 4. 实验计划（建议）

### 阶段 1：可运行验证（1-2 天）
- 实现 SADA-v1，先跑 `ultrafast` 小实验。
- 核对：
  - 训练稳定性（loss 不抖）
  - `s` 分布是否随 epoch 演化
  - ET/WT/TC dice 基本不退化

### 阶段 2：超参扫描（2-3 天）
- `L in {3,5,7}`
- `beta in {0.8,0.9,0.95}`
- warmup `{5,10}`
- 对比基线：固定增强、随机强度增强、SADA

### 阶段 3：主实验（3-5 天）
- 在 `full_liu_mkir_drbd` 或当前主干上跑完整训练。
- 输出：
  - 最优/均值 Dice + HD95
  - 收敛曲线
  - 训练耗时与显存

---

## 5. 风险与规避

1. **多进程 DataLoader 状态同步复杂**
   - 先用 epoch 级 JSON 发布机制规避。

2. **随机裁剪导致 case-level 信号噪声大**
   - 用 EMA + 小窗口抑制噪声；必要时合并最近 K 次观测。

3. **增强强度过大损伤小病灶（ET）**
   - 对 spatial op 设置上限；对 ET-positive case 可加强度上限保护。

