"""
MSFFEC-Net 训练脚本

模型输出:
    pred      : 分割预测图  [B, 1, H, W]
    emb       : 对比学习嵌入 [B, 32, H, W]
    edge_pred : 边缘预测图  [B, 1, H, W]

损失函数:
    L_total = L_seg(DiceBCE) + L_boundary(BCE) + 0.1 * L_NCE(对比损失)
"""

import os
import shutil
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import (
    seeding, create_dir, print_and_save,
    epoch_time, calculate_metrics, ContrastCELoss
)
from utils.metrics import DiceBCELoss
from model import MSFFEC_Net


# =============================================================================
# 数据加载
# =============================================================================

def shuffling(x, y, z):
    x, y, z = shuffle(x, y, z, random_state=42)
    return x, y, z


def load_names(path: str, file_path: str, flag: str):
    """从 txt 文件读取文件名列表，返回 images / masks / edges 路径列表"""
    with open(file_path, "r") as f:
        data = f.read().split("\n")[:-1]
    images = [os.path.join(path, flag, "images", n).replace("\\", "/") for n in data]
    masks  = [os.path.join(path, flag, "masks",  n).replace("\\", "/") for n in data]
    edges  = [os.path.join(path, flag, "edges",  n).replace("\\", "/") for n in data]
    return images, masks, edges


def load_data(path: str, flag: str):
    if flag == "train":
        train_x, train_y, train_e = load_names(path, f"{path}/train.txt", "train")
        valid_x, valid_y, valid_e = load_names(path, f"{path}/val.txt",   "val")
        return (train_x, train_y, train_e), (valid_x, valid_y, valid_e)
    else:
        test_x, test_y, test_e = load_names(path, f"{path}/test.txt", "test")
        return (test_x, test_y, test_e)


# =============================================================================
# Dataset
# =============================================================================

class PolypDataset(Dataset):
    """
    同时加载 image / mask / edge 三通道数据。
    edge 由 Canny 算子预先生成并保存到 edges/ 目录。
    """

    def __init__(self, images_path, masks_path, edges_path,
                 size=(256, 256), transform=None):
        self.images_path = images_path
        self.masks_path  = masks_path
        self.edges_path  = edges_path
        self.transform   = transform
        self.size        = size

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask  = cv2.imread(self.masks_path[index],  cv2.IMREAD_GRAYSCALE)
        edge  = cv2.imread(self.edges_path[index],  cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]
            # edge 与 mask 同步翻转/旋转（使用相同随机状态重新增强）
            aug2  = self.transform(image=image, mask=edge)
            edge  = aug2["mask"]

        h, w = self.size
        image = cv2.resize(image, (w, h))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

        mask = cv2.resize(mask, (w, h))
        mask = np.expand_dims(mask, 0).astype(np.float32) / 255.0

        edge = cv2.resize(edge, (w, h))
        edge = np.expand_dims(edge, 0).astype(np.float32) / 255.0

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            torch.from_numpy(edge),
        )


# =============================================================================
# 训练 / 验证
# =============================================================================

def train_one_epoch(model, loader, optimizer,
                    seg_loss_fn, bce_loss_fn, device):
    """
    一个 epoch 的训练。
    损失: L_total = L_seg + L_boundary + 0.1 * L_NCE
    """
    model.train()

    total_loss = total_seg = total_edge = total_nce = 0.0
    jac = f1 = recall = precision = 0.0

    bar = tqdm(loader, desc="Train", leave=False)
    for x, y, edge in bar:
        x    = x.to(device, dtype=torch.float32)
        y    = y.to(device, dtype=torch.float32)
        edge = edge.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # ── 前向传播 ────────────────────────────────────────────────────────
        pred, emb, edge_pred = model(x)

        # ── 损失计算 ────────────────────────────────────────────────────────
        l_seg      = seg_loss_fn(pred, y)                    # DiceBCE，公式(31)
        l_boundary = bce_loss_fn(edge_pred, edge)            # BCE，公式(7)
        l_nce      = ContrastCELoss()(pred, emb, y)          # 对比损失，公式(21)
        loss       = l_seg + l_boundary + 0.1 * l_nce        # 公式(30)

        loss.backward()
        optimizer.step()

        # ── 指标统计 ────────────────────────────────────────────────────────
        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_edge += l_boundary.item()
        total_nce  += l_nce.item()

        batch_scores = [calculate_metrics(yt, yp) for yt, yp in zip(y, pred)]
        jac       += np.mean([s[0] for s in batch_scores])
        f1        += np.mean([s[1] for s in batch_scores])
        recall    += np.mean([s[2] for s in batch_scores])
        precision += np.mean([s[3] for s in batch_scores])

        bar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return (
        total_loss / n,
        {"seg": total_seg / n, "edge": total_edge / n, "nce": total_nce / n},
        [jac / n, f1 / n, recall / n, precision / n],
    )


@torch.no_grad()
def evaluate(model, loader, seg_loss_fn, bce_loss_fn, device):
    model.eval()

    total_loss = total_seg = total_edge = total_nce = 0.0
    jac = f1 = recall = precision = 0.0

    bar = tqdm(loader, desc="Val  ", leave=False)
    for x, y, edge in bar:
        x    = x.to(device, dtype=torch.float32)
        y    = y.to(device, dtype=torch.float32)
        edge = edge.to(device, dtype=torch.float32)

        pred, emb, edge_pred = model(x)

        l_seg      = seg_loss_fn(pred, y)
        l_boundary = bce_loss_fn(edge_pred, edge)
        l_nce      = ContrastCELoss()(pred, emb, y)
        loss       = l_seg + l_boundary + 0.1 * l_nce

        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_edge += l_boundary.item()
        total_nce  += l_nce.item()

        batch_scores = [calculate_metrics(yt, yp) for yt, yp in zip(y, pred)]
        jac       += np.mean([s[0] for s in batch_scores])
        f1        += np.mean([s[1] for s in batch_scores])
        recall    += np.mean([s[2] for s in batch_scores])
        precision += np.mean([s[3] for s in batch_scores])

    n = len(loader)
    return (
        total_loss / n,
        {"seg": total_seg / n, "edge": total_edge / n, "nce": total_nce / n},
        [jac / n, f1 / n, recall / n, precision / n],
    )


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":

    # ── 超参数 ──────────────────────────────────────────────────────────────
    MODEL_NAME   = "MSFFEC_Net"
    DATASET_NAME = "Kvasir-SEG&CVC-ClinicDB"
    DATA_PATH    = "./Data/TrainDataset/"
    PVT_PATH     = "./backbone/pvt_v2_b2.pth"

    IMAGE_SIZE             = 256
    BATCH_SIZE             = 8
    NUM_EPOCHS             = 200
    LR                     = 1e-4
    EARLY_STOPPING_PATIENCE = 50
    CHANNEL                = 32

    # ── 目录 & 日志 ─────────────────────────────────────────────────────────
    seeding(42)
    save_dir       = f"files/{MODEL_NAME}/{DATASET_NAME}"
    checkpoint_path = f"{save_dir}/checkpoint.pth"
    train_log_path  = f"{save_dir}/train_log.txt"
    create_dir(save_dir)

    if not os.path.exists(train_log_path):
        open(train_log_path, "w").close()

    # 自动备份训练脚本，方便复现
    shutil.copy(os.path.realpath(__file__),
                f"{save_dir}/train_msffec.py")

    writer = SummaryWriter(f"logs/{MODEL_NAME}/{DATASET_NAME}")

    print_and_save(train_log_path, str(datetime.datetime.now()))
    print_and_save(train_log_path,
                   f"Image Size: {IMAGE_SIZE}  Batch: {BATCH_SIZE}  "
                   f"LR: {LR}  Epochs: {NUM_EPOCHS}")

    # ── 数据集 ──────────────────────────────────────────────────────────────
    (train_x, train_y, train_e), (valid_x, valid_y, valid_e) = \
        load_data(DATA_PATH, "train")
    train_x, train_y, train_e = shuffling(train_x, train_y, train_e)

    print_and_save(train_log_path,
                   f"Train: {len(train_x)}  Val: {len(valid_x)}")

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
    ])

    train_loader = DataLoader(
        PolypDataset(train_x, train_y, train_e,
                     size=(IMAGE_SIZE, IMAGE_SIZE), transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
    )
    valid_loader = DataLoader(
        PolypDataset(valid_x, valid_y, valid_e,
                     size=(IMAGE_SIZE, IMAGE_SIZE), transform=None),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── 模型 & 优化器 ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MSFFEC_Net(channel=CHANNEL, pvt_path=PVT_PATH).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5
    )
    seg_loss_fn = DiceBCELoss()                              # L_seg
    bce_loss_fn = nn.BCEWithLogitsLoss()                     # L_boundary

    print_and_save(train_log_path,
                   "Optimizer: Adam | Loss: DiceBCE + BCE(edge) + 0.1*NCE")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    best_f1     = 0.0
    es_count    = 0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_sub, train_metrics = train_one_epoch(
            model, train_loader, optimizer, seg_loss_fn, bce_loss_fn, device
        )
        valid_loss, valid_sub, valid_metrics = evaluate(
            model, valid_loader, seg_loss_fn, bce_loss_fn, device
        )

        scheduler.step(valid_loss)

        # ── TensorBoard ─────────────────────────────────────────────────────
        for tag, val in [("Loss/total",  train_loss),
                          ("Loss/seg",    train_sub["seg"]),
                          ("Loss/edge",   train_sub["edge"]),
                          ("Loss/nce",    train_sub["nce"]),
                          ("Metric/Jaccard",   train_metrics[0]),
                          ("Metric/F1",        train_metrics[1]),
                          ("Metric/Recall",    train_metrics[2]),
                          ("Metric/Precision", train_metrics[3])]:
            writer.add_scalar(f"Train/{tag}", val, epoch)

        for tag, val in [("Loss/total",  valid_loss),
                          ("Loss/seg",    valid_sub["seg"]),
                          ("Loss/edge",   valid_sub["edge"]),
                          ("Loss/nce",    valid_sub["nce"]),
                          ("Metric/Jaccard",   valid_metrics[0]),
                          ("Metric/F1",        valid_metrics[1]),
                          ("Metric/Recall",    valid_metrics[2]),
                          ("Metric/Precision", valid_metrics[3])]:
            writer.add_scalar(f"Valid/{tag}", val, epoch)

        # ── 保存最优模型 ─────────────────────────────────────────────────────
        if valid_metrics[1] > best_f1:
            print_and_save(
                train_log_path,
                f"Val F1 improved {best_f1:.4f} → {valid_metrics[1]:.4f}. "
                f"Saving {checkpoint_path}"
            )
            best_f1  = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            es_count = 0
        else:
            es_count += 1

        mins, secs = epoch_time(t0, time.time())
        log = (
            f"Epoch {epoch+1:03}/{NUM_EPOCHS} | {mins}m {secs}s\n"
            f"  Train | Loss: {train_loss:.4f} "
            f"(seg={train_sub['seg']:.4f}, edge={train_sub['edge']:.4f}, "
            f"nce={train_sub['nce']:.4f}) | "
            f"Jac={train_metrics[0]:.4f} F1={train_metrics[1]:.4f} "
            f"Rec={train_metrics[2]:.4f} Pre={train_metrics[3]:.4f}\n"
            f"  Val   | Loss: {valid_loss:.4f} "
            f"(seg={valid_sub['seg']:.4f}, edge={valid_sub['edge']:.4f}, "
            f"nce={valid_sub['nce']:.4f}) | "
            f"Jac={valid_metrics[0]:.4f} F1={valid_metrics[1]:.4f} "
            f"Rec={valid_metrics[2]:.4f} Pre={valid_metrics[3]:.4f}"
        )
        print_and_save(train_log_path, log)

        # ── Early Stopping ───────────────────────────────────────────────────
        if es_count >= EARLY_STOPPING_PATIENCE:
            print_and_save(
                train_log_path,
                f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} "
                f"epochs without improvement."
            )
            break

    writer.close()
    print_and_save(train_log_path,
                   f"Training finished. Best Val F1: {best_f1:.4f}")