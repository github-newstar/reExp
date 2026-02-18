# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
使用 Swin UNETR 进行 3D 脑肿瘤分割 (BraTS 23 数据集)

本教程使用 Swin UNETR 模型对 BraTS 2023 GLI 挑战数据集进行脑肿瘤分割任务。
数据集路径: /root/autodl-tmp/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
使用 30% 的数据进行训练。

数据说明:
- 模态: MRI (4 种序列)
  - t2f: T2 FLAIR
  - t1c: T1 增强 (T1CE)
  - t1n: T1 原生
  - t2w: T2 加权
- 分割标签: 1=NCR(坏死), 2=ED(水肿), 4=ET(增强肿瘤), 0=背景
- 子区域定义与 BraTS21 相同: ET, TC, WT

注意: BraTS23 文件名格式与 BraTS21 不同:
- BraTS23: BraTS-GLI-xxxxx-xxx-t2f.nii.gz (FLAIR)
- BraTS23: BraTS-GLI-xxxxx-xxx-t1c.nii.gz (T1CE)
- BraTS23: BraTS-GLI-xxxxx-xxx-t1n.nii.gz (T1)
- BraTS23: BraTS-GLI-xxxxx-xxx-t2w.nii.gz (T2)
- BraTS23: BraTS-GLI-xxxxx-xxx-seg.nii.gz (分割标签)
"""

import os
import shutil
import tempfile
import time
import glob
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    MapTransform,
)

from monai.config import print_config
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch
from torch.cuda.amp import autocast, GradScaler
from monai.data import CacheDataset

# 打印 MONAI 配置信息
print_config()

# =============================================================================
# 设置数据目录
# =============================================================================

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"检查点保存目录: {root_dir}")

# =============================================================================
# 设置 BraTS23 数据路径
# =============================================================================

# BraTS23 数据集根目录
brats23_data_dir = "/root/autodl-tmp/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# 数据使用比例 (30%)
DATA_USAGE_RATIO = 0.3

# 日志目录
tensorboard_log_dir = "/root/tf-logs"
os.makedirs(tensorboard_log_dir, exist_ok=True)
print(f"TensorBoard 日志目录: {tensorboard_log_dir}")

# =============================================================================
# 设置平均计量器、数据加载器和检查点保存器
# =============================================================================

class AverageMeter(object):
    """用于计算和存储平均值和当前值的计量器"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # 确保 n 是标量
        if isinstance(n, (list, tuple, np.ndarray)):
            n = np.sum(n)
        self.sum += val * n
        self.count += n
        if isinstance(self.count, np.ndarray):
            self.count = np.sum(self.count)
        if isinstance(self.sum, np.ndarray):
            self.sum = np.sum(self.sum)
        self.avg = self.sum / self.count if self.count > 0 else 0


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    将 BraTS 标签转换为多通道格式（适配 BraTS 2023）
    输入标签: 0=背景, 1=坏死(NCR), 2=水肿(ED), 3=增强肿瘤(ET)
    输出: 3 通道 [TC, WT, ET]
        - TC (肿瘤核心): NCR + ET = 标签 1 或 3
        - WT (全肿瘤): NCR + ED + ET = 标签 1 或 2 或 3
        - ET (增强肿瘤): 仅标签 3
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            
            # 创建 3 个通道
            # 注意: BraTS 2023 使用 3 表示 ET，而不是 4
            tc = (label == 1) | (label == 3)  # 坏死 + 增强
            wt = (label == 1) | (label == 2) | (label == 3)  # 全部肿瘤区域
            et = (label == 3)  # 仅增强
            
            # 堆叠为 [C, H, W, D]
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d


def get_brats23_data_list(data_dir, usage_ratio=0.3, fold=0, n_folds=5, seed=42):
    """
    扫描 BraTS23 数据集目录，构建数据列表
    
    参数:
        data_dir: BraTS23 数据根目录
        usage_ratio: 使用数据的比例 (0.0-1.0)
        fold: 当前验证折编号
        n_folds: 总折数
        seed: 随机种子，保证可复现
    
    返回:
        训练文件列表和验证文件列表
    """
    # 获取所有病例目录
    case_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS-GLI-*")))
    case_dirs = [d for d in case_dirs if os.path.isdir(d)]
    
    print(f"发现 {len(case_dirs)} 个病例文件夹")
    
    # 随机打乱并选择指定比例的数据
    random.seed(seed)
    random.shuffle(case_dirs)
    
    n_selected = max(1, int(len(case_dirs) * usage_ratio))
    selected_cases = case_dirs[:n_selected]
    
    print(f"使用 {n_selected} 个病例 ({usage_ratio*100:.0f}% 的数据)")
    
    # 构建 MONAI 格式的数据列表
    data_list = []
    for case_dir in selected_cases:
        case_name = os.path.basename(case_dir)
        # BraTS23 文件命名格式
        data_list.append({
            "image": [
                os.path.join(case_dir, f"{case_name}-t2f.nii.gz"),  # FLAIR
                os.path.join(case_dir, f"{case_name}-t1c.nii.gz"),  # T1CE
                os.path.join(case_dir, f"{case_name}-t1n.nii.gz"),  # T1
                os.path.join(case_dir, f"{case_name}-t2w.nii.gz"),  # T2
            ],
            "label": os.path.join(case_dir, f"{case_name}-seg.nii.gz"),
        })
    
    # 划分训练集和验证集 (K折交叉验证)
    fold_size = len(data_list) // n_folds
    start_idx = fold * fold_size
    end_idx = start_idx + fold_size if fold < n_folds - 1 else len(data_list)
    
    val_list = data_list[start_idx:end_idx]
    train_list = data_list[:start_idx] + data_list[end_idx:]
    
    print(f"训练集: {len(train_list)} 个病例")
    print(f"验证集: {len(val_list)} 个病例")
    
    return train_list, val_list


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    """保存模型检查点"""
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print(f"保存检查点: {filename}")


# =============================================================================
# 设置数据加载器
# =============================================================================

def get_loader(batch_size, data_dir, fold, roi, usage_ratio=0.3):
    """
    创建训练和验证数据加载器
    
    参数:
        batch_size: 批次大小
        data_dir: 数据目录
        fold: 验证折编号
        roi: 感兴趣区域大小
        usage_ratio: 使用数据的比例
    
    返回:
        训练加载器和验证加载器
    """
    train_files, validation_files = get_brats23_data_list(
        data_dir=data_dir, 
        usage_ratio=usage_ratio,
        fold=fold
    )
    
    # 训练数据转换
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
                allow_smaller=True,
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            # 随机翻转增强
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # 强度归一化和增强
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    
    # 验证数据转换
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    return train_loader, val_loader


# =============================================================================
# 设置超参数
# =============================================================================

# 数据目录配置
data_dir = brats23_data_dir

# 模型和训练超参数
roi = (96, 96, 96)     # 感兴趣区域大小 (减小以适应显存)
batch_size = 2         # 批次大小 (减小以适应显存)
sw_batch_size = 2      # 滑动窗口推理批次大小
fold = 0               # 使用第 0 折进行训练
infer_overlap = 0.5    # 推理重叠率
max_epochs = 100       # 最大训练轮数
val_every = 10         # 每多少轮验证一次

# 创建数据加载器
print("=" * 60)
print("正在创建数据加载器...")
train_loader, val_loader = get_loader(batch_size, data_dir, fold, roi, usage_ratio=DATA_USAGE_RATIO)

# =============================================================================
# 检查数据形状并可视化
# =============================================================================

# 获取第一个训练样本进行可视化
first_train_case = train_loader.dataset.data[0]
img_add = first_train_case["image"][0]  # FLAIR 模态
label_add = first_train_case["label"]

print(f"\n示例图像路径: {img_add}")
print(f"示例标签路径: {label_add}")

img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()
print(f"图像形状: {img.shape}, 标签形状: {label.shape}")

# 找到标签有内容的切片进行可视化
label_sum_z = [label[:, :, z].sum() for z in range(label.shape[2])]
best_slice = label_sum_z.index(max(label_sum_z))
print(f"最佳可视化切片: {best_slice}")

# 可视化示例
plt.figure("数据检查", (18, 6))
plt.subplot(1, 2, 1)
plt.title("图像 (FLAIR)")
plt.imshow(img[:, :, best_slice], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("标签")
plt.imshow(label[:, :, best_slice])
plt.savefig(os.path.join(root_dir, "data_check.png"))
print(f"数据检查图已保存到: {os.path.join(root_dir, 'data_check.png')}")
plt.close()

# =============================================================================
# 创建 Swin UNETR 模型
# =============================================================================

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=4,          # 输入通道数 (4 种 MRI 模态)
    out_channels=3,         # 输出通道数 (3 个子区域)
    feature_size=24,        # 特征大小 (减小以适应显存)
    drop_rate=0.0,          # dropout 率
    attn_drop_rate=0.0,     # 注意力 dropout 率
    dropout_path_rate=0.0,  # 路径 dropout 率
    use_checkpoint=True,    # 使用梯度检查点
).to(device)

print(f"\n模型已创建并移动到设备: {device}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# =============================================================================
# 优化器和损失函数
# =============================================================================

torch.backends.cudnn.benchmark = True

# 清理显存缓存
torch.cuda.empty_cache()
print(f"初始显存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Dice 损失函数
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

# 后处理转换
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)



# 手动计算 Dice 分数的函数
def compute_dice_score(pred, target, smooth=1e-5):
    """
    计算每个类别的 Dice 分数
    
    参数:
        pred: 预测结果 [B, C, H, W, D]，已经过 sigmoid 和阈值处理
        target: 真实标签 [B, C, H, W, D]
        smooth: 平滑因子，避免除零
    
    返回:
        dice_scores: 每个类别的 Dice 分数 [C]
    """
    # 将预测结果二值化（阈值 0.5）
    pred = (pred > 0.5).float()
    
    # 展平空间维度 [B, C, H*W*D]
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    
    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
    pred_sum = pred_flat.sum(dim=2)  # [B, C]
    target_sum = target_flat.sum(dim=2)  # [B, C]
    
    # 计算 Dice: (2*intersection + smooth) / (pred_sum + target_sum + smooth)
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)  # [B, C]
    
    # 返回每个类别的平均 Dice
    return dice.mean(dim=0)  # [C]

# 模型推理器 (使用滑动窗口推理)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

# AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# =============================================================================
# 定义训练和验证轮次
# =============================================================================

def train_epoch(model, loader, optimizer, epoch, loss_func, scaler=None):
    """
    训练一个轮次 - PyTorch Lightning 风格单行刷新，支持混合精度
    """
    model.train()
    epoch_start_time = time.time()
    run_loss = AverageMeter()
    
    # 只在 epoch 开始时清理一次显存
    torch.cuda.empty_cache()
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast():
                logits = model(data)
                loss = loss_func(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        run_loss.update(loss.item(), n=batch_size)
        
        # 计算进度和 ETA
        progress = (idx + 1) / len(loader)
        elapsed = time.time() - epoch_start_time
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        # 单行刷新打印进度
        print(
            f"\rEpoch {epoch:3d}/{max_epochs} │"
            f"{'█' * int(progress * 20):20s}│"
            f" {progress*100:5.1f}% │"
            f" Loss: {run_loss.avg:.4f} │"
            f" {idx+1:4d}/{len(loader)} │"
            f" ETA: {eta:4.0f}s",
            end="",
            flush=True
        )
    
    # 换行，结束本轮进度显示
    total_time = time.time() - epoch_start_time
    print(f" │ 完成 {total_time:.1f}s")
    
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    """
    验证一个轮次 - 使用手动计算 Dice，单行刷新进度
    """
    model.eval()
    val_start_time = time.time()
    
    # 累积所有 batch 的 Dice 分数
    all_dice_tc = []
    all_dice_wt = []
    all_dice_et = []
    
    # 清理显存
    torch.cuda.empty_cache()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].to(device)
            target = batch_data["label"].to(device)
            
            # 推理
            logits = model_inferer(data)
            
            # 后处理: sigmoid -> 阈值化
            prob = post_sigmoid(logits)
            pred = post_pred(prob)
            
            # 计算 Dice 分数 [TC, WT, ET]
            dice_scores = compute_dice_score(pred, target)  # [3]
            
            dice_tc = dice_scores[0].item()
            dice_wt = dice_scores[1].item()
            dice_et = dice_scores[2].item()
            
            all_dice_tc.append(dice_tc)
            all_dice_wt.append(dice_wt)
            all_dice_et.append(dice_et)
            
            # 计算进度
            progress = (idx + 1) / len(loader)
            elapsed = time.time() - val_start_time
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            # 单行刷新打印验证进度
            print(
                f"\rValidation     │"
                f"{'█' * int(progress * 20):20s}│"
                f" {progress*100:5.1f}% │"
                f" TC:{dice_tc:.3f} WT:{dice_wt:.3f} ET:{dice_et:.3f} │"
                f" {idx+1:3d}/{len(loader)} │"
                f" ETA:{eta:4.0f}s",
                end="",
                flush=True
            )
            
    # 计算所有样本的平均 Dice 分数
    mean_dice_tc = np.mean(all_dice_tc)
    mean_dice_wt = np.mean(all_dice_wt)
    mean_dice_et = np.mean(all_dice_et)
    total_time = time.time() - val_start_time
    
    # 换行，显示验证结果
    print(f" │ TC:{mean_dice_tc:.4f} WT:{mean_dice_wt:.4f} ET:{mean_dice_et:.4f}")
    
    # 返回每个类别的平均 Dice 分数 [TC, WT, ET]
    return np.array([mean_dice_tc, mean_dice_wt, mean_dice_et])


# =============================================================================
# 定义训练器
# =============================================================================

def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    log_dir=None,
    scaler=None,
):
    """
    模型训练函数
    """
    # 创建 TensorBoard writer
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志写入: {log_dir}")
    else:
        writer = None
    
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), f"轮次: {epoch}")
        epoch_time = time.time()
        
        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            scaler=scaler,
        )
        print(f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Time={time.time() - epoch_time:.1f}s")

        # 验证
        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            
            print(
                f"Epoch {epoch} Summary: "
                f"Val TC={dice_tc:.4f} WT={dice_wt:.4f} ET={dice_et:.4f} Avg={val_avg_acc:.4f}, "
                f"Time={time.time() - epoch_time:.1f}s"
            )
            
            # 写入 TensorBoard
            if writer is not None:
                writer.add_scalar("Dice/TC", dice_tc, epoch)
                writer.add_scalar("Dice/WT", dice_wt, epoch)
                writer.add_scalar("Dice/ET", dice_et, epoch)
                writer.add_scalar("Dice/Average", val_avg_acc, epoch)
                writer.add_scalar("Loss/Train", train_loss, epoch)
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Learning_Rate", current_lr, epoch)
            
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            
            # 保存最佳模型
            if val_avg_acc > val_acc_max:
                print(f"新的最佳准确率 ({val_acc_max:.6f} --> {val_avg_acc:.6f})")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )
            
            scheduler.step()
    
    print(f"训练完成! 最佳准确率: {val_acc_max:.4f}")
    
    # 关闭 TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard 日志已保存到: {log_dir}")
        print(f"查看命令: tensorboard --logdir={log_dir}")
    
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


# =============================================================================
# 执行训练
# =============================================================================

if __name__ == "__main__":
    start_epoch = 0

    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(tensorboard_log_dir, f"brats23_run_{timestamp}")
    
    # 初始化混合精度训练器
    scaler = GradScaler() if torch.cuda.is_available() else None
    if scaler:
        print("启用混合精度训练 (AMP)")
    
    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        log_dir=run_log_dir,
        scaler=scaler,
    )

    print(f"训练完成，最佳平均 Dice: {val_acc_max:.4f}")

    # =============================================================================
    # 绘制损失和 Dice 指标
    # =============================================================================

    plt.figure("训练", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("轮次平均损失")
    plt.xlabel("轮次")
    plt.plot(trains_epoch, loss_epochs, color="red")
    
    plt.subplot(1, 2, 2)
    plt.title("验证平均 Dice")
    plt.xlabel("轮次")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.savefig(os.path.join(root_dir, "training_curves.png"))
    print(f"训练曲线已保存到: {os.path.join(root_dir, 'training_curves.png')}")
    plt.close()

    plt.figure("详细指标", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("验证 TC Dice")
    plt.xlabel("轮次")
    plt.plot(trains_epoch, dices_tc, color="blue")
    
    plt.subplot(1, 3, 2)
    plt.title("验证 WT Dice")
    plt.xlabel("轮次")
    plt.plot(trains_epoch, dices_wt, color="brown")
    
    plt.subplot(1, 3, 3)
    plt.title("验证 ET Dice")
    plt.xlabel("轮次")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.savefig(os.path.join(root_dir, "dice_details.png"))
    print(f"详细 Dice 曲线已保存到: {os.path.join(root_dir, 'dice_details.png')}")
    plt.close()

    # =============================================================================
    # 测试推理
    # =============================================================================

    # 从验证集中选择一个案例进行推理
    test_case = val_loader.dataset.data[0]
    case_name = os.path.basename(os.path.dirname(test_case["label"]))
    
    print(f"\n测试案例: {case_name}")

    test_files = [test_case]

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = data.Dataset(data=test_files, transform=test_transform)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # =============================================================================
    # 加载最佳保存的检查点并执行推理
    # =============================================================================

    checkpoint_path = os.path.join(root_dir, "model.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True)["state_dict"])
        model.to(device)
        model.eval()
        print(f"已加载检查点: {checkpoint_path}")
    else:
        print("警告: 未找到检查点文件，使用未训练的模型进行测试")
        model.eval()

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for batch_data in test_loader:
            image = batch_data["image"].cuda()
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2  # ED
            seg_out[seg[0] == 1] = 1  # NCR/TC
            seg_out[seg[2] == 1] = 4  # ET

    # =============================================================================
    # 可视化分割输出并与标签比较
    # =============================================================================

    # 加载原始图像进行可视化 (使用 T1CE 模态)
    img_add = test_case["image"][1]  # t1c
    label_add = test_case["label"]
    img = nib.load(img_add).get_fdata()
    label = nib.load(label_add).get_fdata()
    
    # 找到最佳显示切片
    label_sum_z = [label[:, :, z].sum() for z in range(label.shape[2])]
    slice_num = label_sum_z.index(max(label_sum_z))
    
    plt.figure("分割结果", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("图像 (T1CE)")
    plt.imshow(img[:, :, slice_num], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("标签 (Ground Truth)")
    plt.imshow(label[:, :, slice_num])
    plt.subplot(1, 3, 3)
    plt.title("分割结果 (预测)")
    plt.imshow(seg_out[:, :, slice_num])
    plt.savefig(os.path.join(root_dir, "segmentation_result.png"))
    print(f"分割结果图已保存到: {os.path.join(root_dir, 'segmentation_result.png')}")
    plt.close()

    print("\n所有结果已保存到:", root_dir)
