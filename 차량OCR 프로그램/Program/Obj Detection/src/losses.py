# -*- coding: utf-8 -*-
import torch, torch.nn as nn
import torch.nn.functional as F

# ------------ 기존 것들 (그대로 유지) ------------
def bbox_iou_xyxy(a, b, eps=1e-9):
    # a:[N,4], b:[N,4] or [1,4] broadcast
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    bb = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (aa + bb - inter + eps)

class BCEWithLogitsLossWeighted(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    def forward(self, pred, target):
        return self.bce(pred, target)

# ------------ 새로 추가: Focal BCE ------------
class FocalBCE(nn.Module):
    """
    간단·안정 버전의 Focal BCE
    - alpha: 양성/음성 비중
    - gamma: 쉽고 확신 높은 샘플 억제(>0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # logits, target: 동일 shape
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean()

# ------------ 기존: DFL ------------
class DFLLoss(nn.Module):
    """
    간단한 Distribution Focal Loss (classification-style on distance bins)
    reg_logits: [B, 4*bins, H, W]
    targets:    [B, HW, 4]  (stride 정규화된 float distance)
    pos_mask:   [B, HW]     (양성 위치)
    """
    def __init__(self, bins=8):
        super().__init__()
        self.bins = bins

    def forward(self, reg_logits, targets, pos_mask):
        B, C, H, W = reg_logits.shape
        bins = self.bins
        assert C == 4 * bins, "reg channel must be 4*bins"

        # [B,HW,4*bins]
        reg_logits = reg_logits.permute(0,2,3,1).reshape(B, -1, 4*bins)
        # [B,HW,4]
        t = targets
        pm = pos_mask  # [B,HW], bool

        if pm.sum() == 0:
            return reg_logits.sum() * 0.0

        # 정수/소수 분해
        t_clamped = t.clamp(min=0, max=bins-1-1e-4)
        l_id = t_clamped.floor().long()                  # [B,HW,4]
        u_id = (l_id + 1).clamp(max=bins-1)
        w_u = t_clamped - l_id.float()                   # [B,HW,4]
        w_l = 1.0 - w_u

        # gather logits → CE
        reg_logits = reg_logits[pm]                      # [P,4*bins]
        l_id = l_id[pm]                                  # [P,4]
        u_id = u_id[pm]
        w_l = w_l[pm]
        w_u = w_u[pm]

        # 각 좌표별로 softmax+CE
        loss = 0.0
        for i in range(4):
            # [P, bins]
            li = reg_logits[:, i*bins:(i+1)*bins]
            logp = torch.log_softmax(li, dim=1)         # [P,bins]
            loss_l = F.nll_loss(logp, l_id[:, i], reduction='none')
            loss_u = F.nll_loss(logp, u_id[:, i], reduction='none')
            loss += (w_l[:, i] * loss_l + w_u[:, i] * loss_u).mean()
        return loss / 4.0
