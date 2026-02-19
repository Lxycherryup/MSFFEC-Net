"""
MSFFEC-Net: Enhanced Polyp Segmentation via Multi-Scale Feature Fusion with Edge-Aware Enhancement and Contrastive Learning

Modules:
  - GateFusion          : 门控融合模块，用于 EAEM (Edge-Aware Enhance Module)
  - AFIM                : Adaptive Feature Interaction Module
  - ASHFM               : Adaptive Semantic-aware Hierarchical Fusion Module
  - ProjectionHead      : Contrastive Learning Head（对比学习投影头）
  - MSFFEC_Net          : 完整网络
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import pvt_v2_b2


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    """Conv2d + BN (不含 ReLU，保持与原始实现一致)"""

    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# EAEM — Edge-Aware Enhance Module
#   内部使用 GateFusion 对低层特征进行自适应门控融合
# ---------------------------------------------------------------------------

class GateFusion(nn.Module):
    """
    门控融合模块 (GatedFusion)
    对两路特征 x1, x2 计算 softmax 权重后加权融合，
    对应论文 Section 3.2 公式 (3)-(5)。

    Args:
        in_planes (int): 单路输入通道数
    """

    def __init__(self, in_planes: int):
        super().__init__()
        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cat_fea = torch.cat([x1, x2], dim=1)                          # [B, 2C, H, W]
        att_vec = torch.cat([self.gate_1(cat_fea),
                             self.gate_2(cat_fea)], dim=1)             # [B, 2, H, W]
        att_soft = self.softmax(att_vec)                               # W1, W2
        w1, w2 = att_soft[:, 0:1], att_soft[:, 1:2]
        return x1 * w1 + x2 * w2                                      # F_edge，公式(5)


# ---------------------------------------------------------------------------
# AFIM — Adaptive Feature Interaction Module
#   多分支双向特征交互，对应论文 Section 3.3 公式 (8)-(13)
# ---------------------------------------------------------------------------

class AFIM(nn.Module):
    """
    Adaptive Feature Interaction Module
    双分辨率分支（高分辨率 + 低分辨率）通过双向交互融合多尺度特征，
    最终通过残差连接输出，对应论文公式 (8)-(13)。

    Args:
        out_channel (int): 输入/输出通道数（经 Translayer 统一后为 channel）
    """

    def __init__(self, out_channel: int = 32):
        super().__init__()

        # 第一阶段：生成高/低分辨率初始特征（公式8-9-10）
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)          # Down，公式(10)

        self.h2h_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(out_channel)
        self.h2l_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(out_channel)

        # 第二阶段：双向交互（公式11-12）
        self.h2h_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnh_1 = nn.BatchNorm2d(out_channel)
        self.bnl_1 = nn.BatchNorm2d(out_channel)

        # 第三阶段：融合输出（公式13）
        self.h2h_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # 第一阶段：B_i^0, B_i^1（公式9, 10）
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # 第二阶段：双向交互（公式11, 12）
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(F.interpolate(x_l, size=(h, w),
                                          mode='bilinear', align_corners=False))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))   # B̃_i^0
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))   # B̃_i^1

        # 第三阶段：合并 + 残差（公式13）
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(F.interpolate(x_l, size=(h, w),
                                          mode='bilinear', align_corners=False))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x          # F̃̃_i，残差输出


# ---------------------------------------------------------------------------
# ASHFM — Adaptive Semantic-aware Hierarchical Fusion Module
#   自顶向下语义引导融合，对应论文 Section 3.4 公式 (14)-(15)
# ---------------------------------------------------------------------------

class ASHFM(nn.Module):
    """
    Adaptive Semantic-aware Hierarchical Fusion Module
    通过 4×4 转置卷积将高层语义信息以 Hadamard 积形式注入低层特征，
    两级级联实现 F̃̃4 → F̃̃3 → F̃̃2 的自顶向下语义引导，
    对应论文公式 (14)-(15)。

    Args:
        out_channel (int): 各层统一通道数
    """

    def __init__(self, out_channel: int = 32):
        super().__init__()
        # TC_{4×4}，公式(14)(15) 中的转置卷积
        self.up_4_to_3 = nn.ConvTranspose2d(
            out_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn_4_to_3 = nn.BatchNorm2d(out_channel)

        self.up_3_to_2 = nn.ConvTranspose2d(
            out_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn_3_to_2 = nn.BatchNorm2d(out_channel)

    def forward(self,
                f4: torch.Tensor,
                f3: torch.Tensor,
                f2: torch.Tensor):
        """
        Args:
            f4: F̃̃_4，最高层特征
            f3: F̃̃_3
            f2: F̃̃_2，最低层特征
        Returns:
            f3_out: F̃̃_3'（公式14）
            f2_out: F̃̃_2'（公式15）
        """
        # 公式(14): F̃̃_3' = TC(BN(F̃̃_4)) × F̃̃_3 + F̃̃_3
        sem_4 = self.bn_4_to_3(self.up_4_to_3(f4))   # 上采样至 f3 分辨率
        f3_out = sem_4 * f3 + f3

        # 公式(15): F̃̃_2' = TC(BN(F̃̃_3')) × F̃̃_2 + F̃̃_2
        sem_3 = self.bn_3_to_2(self.up_3_to_2(f3_out))
        f2_out = sem_3 * f2 + f2

        return f3_out, f2_out


# ---------------------------------------------------------------------------
# Contrastive Learning Head — 对比学习投影头
#   对应论文 Section 3.5 公式 (16)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Contrastive Learning Projection Head
    三层转置卷积逐步恢复空间分辨率，最终输出 L2 归一化的像素级嵌入，
    对应论文公式 (16)：E = Up²(Conv1×1(TCBR³(F4)))

    Args:
        dim (int): 输入/输出嵌入维度，默认 32
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        self.pro = nn.Sequential(
            # TCBR × 3（3 次转置卷积，步长2，分辨率 ×8）
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            # Conv1×1 通道调整
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.pro(x)
        # Up²：2× 双线性上采样（合计输出为原图分辨率）
        proj = F.interpolate(proj, scale_factor=2,
                             mode='bilinear', align_corners=False)
        return F.normalize(proj, p=2, dim=1)   # L2 归一化


# ---------------------------------------------------------------------------
# MSFFEC-Net — 完整网络
# ---------------------------------------------------------------------------

class MSFFEC_Net(nn.Module):
    """
   
    Encoder : PVTv2-B2（预训练）
    Decoder :
      ├─ EAEM  : Edge-Aware Enhance Module（GateFusion + 上采样分支）
      ├─ AFIM  : Adaptive Feature Interaction Module（×3，作用于 F2/F3/F4）
      ├─ ASHFM : Adaptive Semantic-aware Hierarchical Fusion Module
      └─ CL Head : ProjectionHead（对比学习，仅训练阶段使用）

    Args:
        channel   (int) : 统一特征维度，默认 32
        pvt_path  (str) : PVTv2-B2 预训练权重路径
    """

    def __init__(self, channel: int = 32, pvt_path: str = 'pvt_v2_b2.pth'):
        super().__init__()

        # ── Encoder: PVTv2-B2 ──────────────────────────────────────────────
        self.backbone = pvt_v2_b2()
        state_dict = {
            k: v for k, v in torch.load(pvt_path, map_location='cpu').items()
            if k in self.backbone.state_dict()
        }
        self.backbone.load_state_dict(state_dict, strict=False)

        # 通道对齐：将各阶段输出统一压缩到 channel 维
        self.trans1 = BasicConv2d(64,  channel, 1)   # F1: H/4
        self.trans2 = BasicConv2d(128, channel, 1)   # F2: H/8
        self.trans3 = BasicConv2d(320, channel, 1)   # F3: H/16
        self.trans4 = BasicConv2d(512, channel, 1)   # F4: H/32

        # ── EAEM: Edge-Aware Enhance Module ───────────────────────────────
        # GateFusion 融合 F1, F2 低层特征
        self.eaem_gate   = GateFusion(channel)
        # 两次 CBR + Up 生成边缘预测图
        self.eaem_conv0  = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
        )
        self.eaem_conv1  = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
        )
        self.eaem_out    = nn.Conv2d(channel, 1, 1)   # 边缘预测，L_boundary
        self.up2         = nn.Upsample(scale_factor=2, mode='bilinear',
                                       align_corners=True)

        # ── AFIM: Adaptive Feature Interaction Module ──────────────────────
        # 分别作用于 F2, F3, F4（F1 直接用于边缘分支）
        self.afim2 = AFIM(channel)
        self.afim3 = AFIM(channel)
        self.afim4 = AFIM(channel)

        # ── ASHFM: Adaptive Semantic-aware Hierarchical Fusion Module ──────
        self.ashfm = ASHFM(channel)

        # ── Contrastive Learning Head ──────────────────────────────────────
        self.contrast_head = ProjectionHead(dim=channel)

        # ── 解码器上采样 + 最终预测 ─────────────────────────────────────────
        # 将 ASHFM 输出的 F2' 上采样后与 F1 融合，再预测
        self.decoder_up = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=4,
                               stride=2, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.prediction = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1)
        )

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入图像，shape = [B, 3, H, W]，H=W=256

        Returns (训练阶段):
            pred      : 分割预测图，[B, 1, H, W]
            emb       : 对比学习嵌入，[B, 32, H, W]，L2 归一化
            edge_pred : 边缘预测图，[B, 1, H, W]

        Returns (推理阶段仅需 pred):
            pred
        """
        B, C, H, W = x.shape

        # ── Encoder ─────────────────────────────────────────────────────────
        pvt_feats = self.backbone(x)
        f1 = pvt_feats[0]   # [B, 64,  H/4,  W/4]
        f2 = pvt_feats[1]   # [B, 128, H/8,  W/8]
        f3 = pvt_feats[2]   # [B, 320, H/16, W/16]
        f4 = pvt_feats[3]   # [B, 512, H/32, W/32]

        # 通道统一
        f1 = self.trans1(f1)   # [B, 32, H/4,  W/4]
        f2 = self.trans2(f2)   # [B, 32, H/8,  W/8]
        f3 = self.trans3(f3)   # [B, 32, H/16, W/16]
        f4 = self.trans4(f4)   # [B, 32, H/32, W/32]

        # ── Contrastive Learning Head（作用于最深层 F4）──────────────────────
        emb = self.contrast_head(f4)   # [B, 32, H, W]

        # ── EAEM: 低层特征 F1, F2 → 边缘预测 ───────────────────────────────
        f2_up = F.interpolate(f2, size=f1.shape[2:],
                              mode='bilinear', align_corners=False)
        f_edge = self.eaem_gate(f1, f2_up)              # GateFusion，公式(3)-(5)
        e0 = self.eaem_conv0(f_edge)                    # [B, 32, H/4, W/4]
        e1 = self.eaem_conv1(self.up2(e0))              # [B, 32, H/2, W/2]
        edge_pred = self.eaem_out(self.up2(e1))         # [B, 1,  H,   W]，L_boundary

        # ── AFIM: 各层独立多尺度交互（公式8-13）────────────────────────────
        f2 = self.afim2(f2)   # F̃̃_2
        f3 = self.afim3(f3)   # F̃̃_3
        f4 = self.afim4(f4)   # F̃̃_4

        # ── ASHFM: 自顶向下语义融合（公式14-15）────────────────────────────
        _, f2_fused = self.ashfm(f4, f3, f2)   # 取 F̃̃_2'

        # ── Decoder: F2' 上采样 + F1 残差 → 最终分割预测 ───────────────────
        f1_dec = f1 + self.decoder_up(f2_fused)   # [B, 32, H/4, W/4]
        pred_low = self.prediction(f1_dec)         # [B, 1,  H/4, W/4]
        pred = F.interpolate(pred_low, scale_factor=4,
                             mode='bilinear', align_corners=False)  # [B, 1, H, W]

        return pred, emb, edge_pred


# ---------------------------------------------------------------------------
# 快速验证
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from thop import profile

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 3, 256, 256).to(device)

    model = MSFFEC_Net(channel=32, pvt_path='/home/lxy/MSFFEC-Net/backbone/pvt_v2_b2.pth').to(device)
    model.eval()

    with torch.no_grad():
        pred, emb, edge = model(x)

    print(f"Segmentation output : {pred.shape}")    # [2, 1, 256, 256]
    print(f"Contrastive embedding: {emb.shape}")    # [2, 32, 256, 256]
    print(f"Edge prediction      : {edge.shape}")   # [2, 1, 256, 256]

    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs : {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")