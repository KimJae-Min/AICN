import torch, torch.nn as nn
import torch.nn.functional as F

def bbox_iou_xyxy(boxes1, boxes2, eps=1e-9):
    # boxes1: [N,4], boxes2: [M,4] or [4]
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    x1 = torch.max(boxes1[..., 0:1], boxes2[..., 0:1])
    y1 = torch.max(boxes1[..., 1:2], boxes2[..., 1:2])
    x2 = torch.min(boxes1[..., 2:3], boxes2[..., 2:3])
    y2 = torch.min(boxes1[..., 3:4], boxes2[..., 3:4])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    a2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + eps)

class BCEWithLogitsLossWeighted(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        # buffer로 등록해 저장/로드는 되되, forward 때 pred.device로 옮길 수 있게
        self.register_buffer("pos_weight_buf", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, pred, target):
        pw = self.pos_weight_buf.to(pred.device)
        return F.binary_cross_entropy_with_logits(pred, target.to(pred.device), pos_weight=pw, reduction='none').mean()

class DFLLoss(nn.Module):
    def __init__(self, bins=16):
        super().__init__()
        self.bins = bins

    @torch.no_grad()
    def build_soft_targets(self, dist):  # dist: [P,4] in [0, bins-1]
        # 선형 보간 : floor/ceil에 (1-d), d 가중치
        lb = dist.floor().clamp(min=0, max=self.bins-1)
        ub = (lb + 1).clamp(max=self.bins-1)
        d  = (dist - lb).clamp(0,1)
        lb = lb.long(); ub = ub.long()

        # one-hot 2개를 합치는 방식으로 [P,4,bins] 분포 생성
        P = dist.shape[0]
        tgt = torch.zeros(P, 4, self.bins, device=dist.device, dtype=torch.float32)
        tgt.scatter_(2, lb.unsqueeze(-1), (1.0 - d).unsqueeze(-1))
        # ub==lb인 경우도 있으니 add_ 사용
        tgt.scatter_add_(2, ub.unsqueeze(-1), d.unsqueeze(-1))
        return tgt  # [P,4,bins]

    def forward(self, reg_logits, dist_targets, pos_mask):
        """
        reg_logits: [B, 4*bins, H, W]
        dist_targets: [B, HW, 4]  (stride로 나눈 거리값)
        pos_mask: [B, HW] bool
        """
        B, _, H, W = reg_logits.shape
        bins = self.bins
        HW = H * W

        # [B,HW,4*bins] -> [B,HW,4,bins]
        logits = reg_logits.permute(0,2,3,1).reshape(B, HW, 4, bins)
        # pos만 사용
        pm = pos_mask.view(B, HW)
        if pm.sum() == 0:
            return logits.sum() * 0.0

        logits_pos = logits[pm]                 # [P,4,bins]
        dist_pos   = dist_targets[pm]           # [P,4]

        # [0, bins-1]로 클램프
        dist_pos = dist_pos.clamp(0, bins-1)
        tgt = self.build_soft_targets(dist_pos) # [P,4,bins]

        # Cross-Entropy with soft targets = -sum(tgt * log_softmax)
        logp = F.log_softmax(logits_pos, dim=-1)
        loss = -(tgt * logp).sum(dim=-1).mean() # 평균
        return loss
