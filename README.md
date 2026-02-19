# MSFFEC-Net

**MSFFEC-Net: Enhanced Polyp Segmentation via Multi-Scale Feature Fusion with Edge-Aware Enhancement and Contrastive Learning**

MSFFEC-Net 是一个用于息肉图像分割的深度学习网络。它基于 PVTv2-B2 编码器，结合边缘感知增强、多尺度特征融合和对比学习三大核心模块，实现高精度的息肉区域分割。

> 📄 **论文投稿至 *The Visual Computer*（Springer）**

---

## 网络结构

<img width="1442" height="747" alt="image" src="https://github.com/user-attachments/assets/0e7a7418-6560-4f0a-ade0-f7c1f956065f" />


MSFFEC-Net 由以下四个核心模块组成：

| 模块 | 全称 | 作用 |
|------|------|------|
| **EAEM** | Edge-Aware Enhance Module | 融合低层特征，生成精确边缘预测，抑制模糊边界 |
| **AFIM** | Adaptive Feature Interaction Module | 双分辨率分支双向交互，处理不同尺寸息肉 |
| **ASHFM** | Adaptive Semantic-aware Hierarchical Fusion Module | 自顶向下语义引导，防止高层语义在传播中衰减 |
| **CL Head** | Contrastive Learning Head | 跨图像像素级对比学习，增强小息肉特征表达 |

---

## 项目结构

```
MSFFEC-Net/
├── backbone/
│   ├── pvtv2.py              # PVTv2 编码器定义
│   ├── pvt_v2_b2.pth         # PVTv2-B2 预训练权重
│   └── __init__.py
├── model/
│   ├── ours.py               # MSFFEC-Net 完整网络定义
│   └── __init__.py
├── utils/
│   ├── utils.py              # 工具函数（seeding, epoch_time 等）
│   ├── metrics.py            # 损失函数（DiceBCELoss）& 评估指标
│   ├── ContrastCELoss.py     # 像素级对比学习损失（L_NCE）
│   └── __init__.py
├── Data/
│   ├── TrainDataset/
│   │   ├── train/            # 训练集（images/ masks/ edges/）
│   │   ├── val/              # 验证集（images/ masks/ edges/）
│   │   ├── train.txt         # 训练集文件名列表
│   │   ├── val.txt           # 验证集文件名列表
│   │   ├── split_data.py     # 数据集划分脚本
│   │   └── get_edges.py      # Canny 边缘图批量生成脚本
│   └── TestDataset/
│       ├── Kvasir-SEG/
│       ├── CVC-ClinicDB/
│       ├── CVC-ColonDB/
│       ├── CVC-300/
│       ├── ETIS-LaribPolypDB/
│       └── private_dataset/
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
├── files/                    # 训练日志 & checkpoint 保存目录
├── results/                  # 测试结果保存目录
└── logs/                     # TensorBoard 日志目录
```

---

## 环境依赖

- Python 3.8+
- PyTorch 2.0+ & CUDA 11.8+
- timm
- albumentations
- opencv-python
- scikit-learn
- tqdm
- tensorboard
- thop（可选，用于计算 FLOPs & Params）

```bash
pip install torch torchvision timm albumentations opencv-python scikit-learn tqdm tensorboard thop
```

---

## 数据准备

### 1. 公开数据集下载

本项目使用以下五个公开息肉分割数据集进行训练和评估：

| 数据集 | 图像数量 | 下载地址 |
|--------|---------|---------|
| Kvasir-SEG | 1000 | [link](https://datasets.simula.no/kvasir-seg/) |
| CVC-ClinicDB | 612 | [link](https://polyp.grand-challenge.org/) |
| CVC-ColonDB | 380 | [link](https://figshare.com/articles/dataset/CVC-ColonDB/15083958) |
| CVC-300 | 60 | [link](https://figshare.com/articles/dataset/CVC-300/15083870) |
| ETIS-LaribPolypDB | 196 | [link](https://polyp.grand-challenge.org/) |

本文使用的所有数据集可以通过此链接获得：https://pan.baidu.com/s/1cg6IQ9R7F3l98BuNidcmtQ?pwd=cyam 提取码: cyam 


训练集：从 Kvasir-SEG 和 CVC-ClinicDB 中随机抽取共 1612 张，按 8:1:1 划分为训练/验证/测试集。

### 2. 目录格式

每个数据集子目录需包含三个文件夹：

```
<dataset>/
├── images/    # 原始内镜图像（.png / .jpg）
├── masks/     # 分割标注（二值图，像素值 0 或 255）
└── edges/     # 边缘标注（由 Canny 算子预先生成，见步骤 3）
```



### 3. 生成文件名列表

```bash
cd Data/TrainDataset
python get_names.py   # 生成 train.txt 和 val.txt
```

`train.txt` / `val.txt` 每行为一个文件名（不含路径），例如：

```
cjc1aa3z7djcq0850g3t4qlvy.jpg
cjc1abiw3djdm0850htpchkev.jpg
...
```

---

## 训练

```bash
python train.py
```

**主要超参数**（在 `train.py` 头部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMAGE_SIZE` | 256 | 输入图像分辨率 |
| `BATCH_SIZE` | 8 | 批大小 |
| `NUM_EPOCHS` | 200 | 最大训练轮数 |
| `LR` | 1e-4 | Adam 初始学习率 |
| `CHANNEL` | 32 | Decoder 统一特征维度 |
| `EARLY_STOPPING_PATIENCE` | 50 | 连续无提升轮数触发早停 |

**损失函数**：

$$L_{total} = L_{seg} + L_{boundary} +  L_{NCE}$$

- $L_{seg}$：DiceLoss + 加权 BCE（分割主损失）
- $L_{boundary}$：BCE（边缘预测监督）
- $L_{NCE}$：跨图像像素级对比损失

**训练产出**：

```
files/MSFFEC_Net/<数据集名>/
├── checkpoint.pth    # 最优模型权重（Val Dice 最高时保存）
└── train_log.txt     # 逐 epoch 训练日志
```

**TensorBoard 可视化**：

```bash
tensorboard --logdir logs
```

---

## 测试

```bash
python test.py
```

测试脚本将自动对以下 6 个数据集逐一推理：

- Kvasir-SEG / CVC-ClinicDB（训练集内测试）
- CVC-ColonDB / CVC-300 / ETIS-LaribPolypDB（泛化性测试）
- private_dataset（私有临床数据集，可选）

**测试产出**：

```
results/MSFFEC_Net/<数据集>/
├── mask/     # 二值化预测掩码（PNG）
└── joint/    # 原图 | GT | 预测 横向拼接图（JPG）

files/MSFFEC_Net/test_result.txt   # 所有数据集指标汇总
```

终端输出示例：

```
Dataset                   Jaccard     Dice   Recall     Prec      Acc       F2     
--------------------------------------------------------------------------------
Kvasir-SEG                 0.8680   0.9220   0.9130   0.9470   0.9810   0.9150    
CVC-ClinicDB               0.8980   0.9450   0.9350   0.9580   0.9930   0.9380    
CVC-ColonDB                0.7110   0.7980   0.7880   0.8550   0.9690   0.7850    
CVC-300                    0.8210   0.8930   0.9230   0.8810   0.9940   0.9070    
ETIS-LaribPolypDB          0.6690   0.7440   0.7980   0.8000   0.9850   0.7650   
```

---

## 评估指标

| 指标 | 说明 |
|------|------|
| **Jaccard (IoU)** | 预测与 GT 的交并比 |
| **Dice (F1)** | Dice 相似系数，主要评估指标 |
| **Recall** | 召回率，衡量漏检率 |
| **Precision** | 精确率，衡量误检率 |
| **Accuracy** | 像素级分类准确率 |
| **F2** | F2-Score，比 F1 更侧重召回率，适合临床场景 |
| **FPS** | 推理帧率（单张图像，GPU） |

---

## 主要实验结果

与 SOTA 方法在 CVC-ClinicDB 上的对比（训练集内）：

| 方法 | Jaccard | Dice | FPS |
|------|---------|------|-----|
| U-Net | 0.808 | 0.869 | 43.12 |
| PraNet | 0.859 | 0.914 | 42.30 |
| Polyp-PVT | 0.877 | 0.928 | 50.64 |
| CTNet | 0.871 | 0.919 | 33.30 |
| **MSFFEC-Net (Ours)** | **0.898** | **0.945** | **50.60** |

完整结果见论文 Table 2–6。



## 致谢

感谢上海光华中西医结合医院提供私有临床数据集支持（伦理批准号：2023-K-90）。本研究受国家自然科学基金（61801288）及国家重点临床专科学科建设项目（Z155080000004）资助。
