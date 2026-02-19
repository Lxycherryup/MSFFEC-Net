"""
MSFFEC-Net 测试脚本

模型输出:
    pred      : 分割预测图  [B, 1, H, W]
    emb       : 对比学习嵌入 [B, 32, H, W]  
    edge_pred : 边缘预测图  [B, 1, H, W]   

保存内容:
    results/{MODEL_NAME}/{dataset}/mask/   : 二值化预测掩码
    results/{MODEL_NAME}/{dataset}/joint/  : 原图 | GT | 预测 拼接图
    files/{MODEL_NAME}/test_result.txt     : 各数据集指标汇总
"""

import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from operator import add
import numpy as np
import cv2
import torch
from tqdm import tqdm

from utils import create_dir, seeding, calculate_metrics
from train import load_data          # 复用训练脚本中的 load_data
from model import MSFFEC_Net


# =============================================================================
# 单个数据集评估
# =============================================================================

def evaluate(model, save_path, test_x, test_y, test_edge,
             size, model_name, test_dataset_name, device):
    """
    逐图推理、计算指标并保存可视化结果。

    Args:
        model            : 已加载权重的 MSFFEC_Net（eval 模式）
        save_path        : 结果保存根目录
        test_x/y/edge    : 图像 / mask / edge 路径列表
        size             : (H, W)，默认 (256, 256)
        model_name       : 用于写入结果文件的模型标识
        test_dataset_name: 数据集名称
        device           : torch.device
    """
    metrics_score = [0.0] * 7   # jaccard, f1, recall, precision, acc, f2, _
    time_taken    = []

    for x_path, y_path, _ in tqdm(
            zip(test_x, test_y, test_edge),
            total=len(test_x),
            desc=f"  {test_dataset_name}"):

        stem = os.path.splitext(os.path.basename(y_path))[0]

        # ── 读取 & 预处理图像 ────────────────────────────────────────────────
        image_bgr = cv2.imread(x_path, cv2.IMREAD_COLOR)
        image_bgr = cv2.resize(image_bgr, (size[1], size[0]))
        save_img  = image_bgr.copy()

        image = np.transpose(image_bgr, (2, 0, 1)).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # ── 读取 & 预处理 mask ──────────────────────────────────────────────
        mask_gray = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        mask_gray = cv2.resize(mask_gray, (size[1], size[0]))
        save_mask = np.stack([mask_gray] * 3, axis=-1)          # BGR 展示用

        mask_t = torch.from_numpy(
            mask_gray[None, None].astype(np.float32) / 255.0
        ).to(device)

        # ── 推理 ────────────────────────────────────────────────────────────
        with torch.no_grad():
            t0 = time.time()
            pred, _, _ = model(image)          # 只取分割预测，忽略 emb & edge
            pred = torch.sigmoid(pred)
            time_taken.append(time.time() - t0)

        # ── 指标累加 ────────────────────────────────────────────────────────
        score = calculate_metrics(mask_t, pred)
        metrics_score = list(map(add, metrics_score, score))

        # ── 生成可视化预测图 ─────────────────────────────────────────────────
        pred_np = pred[0, 0].cpu().numpy()          # [H, W]
        pred_bin = ((pred_np > 0.5).astype(np.uint8) * 255)
        pred_vis = np.stack([pred_bin] * 3, axis=-1)

        # ── 保存单张 mask ────────────────────────────────────────────────────
        cv2.imwrite(f"{save_path}/mask/{stem}.png", pred_bin)

        # ── 保存拼接图：原图 | GT | 预测 ─────────────────────────────────────
        divider = np.ones((size[0], 10, 3), dtype=np.uint8) * 255
        joint   = np.concatenate(
            [save_img, divider, save_mask, divider, pred_vis], axis=1
        )
        cv2.imwrite(f"{save_path}/joint/{stem}.jpg", joint)

    # ── 汇总指标 ─────────────────────────────────────────────────────────────
    n         = len(test_x)
    jaccard   = metrics_score[0] / n
    f1        = metrics_score[1] / n
    recall    = metrics_score[2] / n
    precision = metrics_score[3] / n
    acc       = metrics_score[4] / n
    f2        = metrics_score[5] / n
    mean_fps  = 1.0 / np.mean(time_taken)

    result_line = (
        f"[{test_dataset_name}] "
        f"Jaccard={jaccard:.4f}  Dice={f1:.4f}  "
        f"Recall={recall:.4f}  Precision={precision:.4f}  "
        f"Acc={acc:.4f}  F2={f2:.4f}  "
        f"FPS={mean_fps:.2f}"
    )
    print(result_line)

    result_file = f"./files/{model_name}/test_result.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "a") as f:
        f.write(result_line + "\n")

    return {
        "jaccard": jaccard, "f1": f1, "recall": recall,
        "precision": precision, "acc": acc, "f2": f2, "fps": mean_fps,
    }


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    seeding(42)

    # ── 配置 ─────────────────────────────────────────────────────────────────
    MODEL_NAME   = "MSFFEC_Net"
    DATASET_NAME = "Kvasir-SEG&CVC-ClinicDB"   # 训练时使用的数据集（用于定位 checkpoint）
    PVT_PATH     = "./backbone/pvt_v2_b2.pth"
    CHANNEL      = 32
    SIZE         = (256, 256)

    CHECKPOINT_PATH = f"./files/{MODEL_NAME}/{DATASET_NAME}/checkpoint.pth"

    TEST_DATASETS = [
        "Kvasir-SEG",
        "CVC-ClinicDB",
        "CVC-ColonDB",
        "CVC-300",
        "ETIS-LaribPolypDB",
        "private_dataset",
    ]

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MSFFEC_Net(channel=CHANNEL, pvt_path=PVT_PATH).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # 清空上次的结果文件，避免追加混乱
    result_file = f"./files/{MODEL_NAME}/test_result.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        f.write(f"Model: {MODEL_NAME} | Checkpoint: {CHECKPOINT_PATH}\n")
        f.write("=" * 80 + "\n")

    # ── 逐数据集测试 ──────────────────────────────────────────────────────────
    all_results = {}
    for dataset_name in TEST_DATASETS:
        data_path = f"./Data/TestDataset/{dataset_name}"
        if not os.path.exists(data_path):
            print(f"[SKIP] {dataset_name} not found at {data_path}")
            continue

        test_x, test_y, test_edge = load_data(data_path, "test")
        test_x    = sorted(test_x)
        test_y    = sorted(test_y)
        test_edge = sorted(test_edge)

        # 创建保存目录
        save_path = f"./results/{MODEL_NAME}/{dataset_name}"
        for sub in ["mask", "joint"]:
            create_dir(f"{save_path}/{sub}")

        metrics = evaluate(
            model, save_path,
            test_x, test_y, test_edge,
            SIZE, MODEL_NAME, dataset_name, device
        )
        all_results[dataset_name] = metrics

    # ── 打印汇总表格 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Dataset':<25} {'Jaccard':>8} {'Dice':>8} {'Recall':>8} "
          f"{'Prec':>8} {'Acc':>8} {'F2':>8} {'FPS':>8}")
    print("-" * 80)
    for ds, m in all_results.items():
        print(f"{ds:<25} {m['jaccard']:>8.4f} {m['f1']:>8.4f} {m['recall']:>8.4f} "
              f"{m['precision']:>8.4f} {m['acc']:>8.4f} {m['f2']:>8.4f} {m['fps']:>8.2f}")
    print("=" * 80)

    # 汇总表也写入结果文件
    with open(result_file, "a") as f:
        f.write("\nSummary\n" + "=" * 80 + "\n")
        f.write(f"{'Dataset':<25} {'Jaccard':>8} {'Dice':>8} {'Recall':>8} "
                f"{'Prec':>8} {'Acc':>8} {'F2':>8} {'FPS':>8}\n")
        f.write("-" * 80 + "\n")
        for ds, m in all_results.items():
            f.write(
                f"{ds:<25} {m['jaccard']:>8.4f} {m['f1']:>8.4f} "
                f"{m['recall']:>8.4f} {m['precision']:>8.4f} "
                f"{m['acc']:>8.4f} {m['f2']:>8.4f} {m['fps']:>8.2f}\n"
            )