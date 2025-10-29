# -*- coding: utf-8 -*-
import os, random, time, csv, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from .model import LP_YOLO_Fuse
from .losses import bbox_iou_xyxy, BCEWithLogitsLossWeighted, DFLLoss
from .utils_common import dist2bbox_ltbr

# =========================
# CLI / Config
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"C:/Users/win/Desktop/lp_yolo_fuse/data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--use_psa", type=int, default=0)     # 1:on, 0:off
    ap.add_argument("--use_dfl", type=int, default=0)     # 1:on, 0:off
    ap.add_argument("--bins", type=int, default=8)        # DFL bins
    ap.add_argument("--dfl_weight", type=float, default=0.5)

    ap.add_argument("--warmup_steps", type=int, default=0)

    ap.add_argument("--pos_expand", type=float, default=0.25)  # 양성영역 확장(stride*x)
    ap.add_argument("--assign_topk", type=int, default=3)      # GT당 top-k 포인트
    ap.add_argument("--letterbox", type=int, default=0)        # 1: keep aspect + pad, 0: plain resize

    ap.add_argument("--run", default="default")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_dir", default="")                  # 기본: DATA_ROOT.parent

    return ap.parse_args()

# =========================
# Utils
# =========================
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_yolo_txt(txt_path: Path) -> torch.Tensor:
    """
    YOLO txt → [N,4] (정규 xyxy)
    """
    if not txt_path.exists():
        return torch.zeros(0, 4, dtype=torch.float32)
    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    boxes = []
    for l in lines:
        parts = l.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)

def letterbox_resize(im: Image.Image, out_size: int):
    """
    keep-aspect resize + pad to square out_size
    Returns: np.float32 [H,W,3] in [0,1], and scale/offset for box transform
    """
    w0, h0 = im.size
    scale = min(out_size / w0, out_size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    im_r = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (out_size, out_size), (114, 114, 114))
    dx = (out_size - nw) // 2
    dy = (out_size - nh) // 2
    canvas.paste(im_r, (dx, dy))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr, scale, dx, dy, (w0, h0)

def load_image_and_scale_boxes(img_path: Path, boxes_xyxy_norm: torch.Tensor, out_size: int, letterbox: bool):
    """
    이미지 -> [3,H,W] (0~1)
    라벨: 정규좌표 → px → 리사이즈/패딩 변환된 px로 변환
    """
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    if letterbox:
        arr, scale, dx, dy, (W0, H0) = letterbox_resize(im, out_size)
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if boxes_xyxy_norm.numel():
            xyxy = boxes_xyxy_norm.clone()
            # norm -> px in orig
            xyxy[:, [0, 2]] *= W0
            xyxy[:, [1, 3]] *= H0
            # scale & pad
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scale + dx
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scale + dy
        else:
            xyxy = torch.zeros(0, 4, dtype=torch.float32)
    else:
        im_r = im.resize((out_size, out_size), Image.BILINEAR)
        arr = np.array(im_r, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if boxes_xyxy_norm.numel():
            xyxy = boxes_xyxy_norm.clone()
            xyxy[:, [0, 2]] *= W
            xyxy[:, [1, 3]] *= H
            sx = out_size / float(W); sy = out_size / float(H)
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy
        else:
            xyxy = torch.zeros(0, 4, dtype=torch.float32)
    return img, xyxy

# =========================
# Dataset / Dataloader
# =========================
class YOLOTxtPlateDataset(Dataset):
    def __init__(self, img_dir: Path, lbl_dir: Path, img_size: int, letterbox: bool):
        self.imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.letterbox = letterbox

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        ip = self.imgs[idx]
        lp = self.lbl_dir / f"{ip.stem}.txt"
        boxes_norm = read_yolo_txt(lp)  # [N,4] in [0,1]
        img, boxes = load_image_and_scale_boxes(ip, boxes_norm, self.img_size, letterbox=self.letterbox)
        labels = torch.zeros((boxes.shape[0],), dtype=torch.long)  # 단일 클래스
        return img, boxes, labels

def collate_fn(batch):
    imgs, boxes_list, labels_list = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(boxes_list), list(labels_list)

# =========================
# Grid & target
# =========================
def build_grid(H, W, stride, device):
    ys = torch.arange(H, device=device) + 0.5
    xs = torch.arange(W, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    return cx * stride, cy * stride  # pixel center

def make_targets(
    boxes_list: List[torch.Tensor],
    cx: torch.Tensor, cy: torch.Tensor,
    B: int, stride: int,
    pos_expand: float = 0.25,
    assign_topk: int = 3
):
    """
    GT 사각형을 좌우상하로 stride*pos_expand 만큼 확장.
    각 GT마다 center 거리 기준 top-k 포인트를 강제 양성.
    """
    H, W = cy.shape[0], cx.shape[1]
    obj_tgt = torch.zeros((B, 1, H, W), device=cx.device, dtype=torch.float32)
    pos_mask_flat = torch.zeros((B, H * W), device=cx.device, dtype=torch.bool)

    cxv, cyv = cx.view(-1), cy.view(-1)  # [HW]
    expand = stride * float(pos_expand)
    k = max(1, int(assign_topk))

    for b in range(B):
        boxes = boxes_list[b].to(cx.device)  # [N,4]
        if boxes.numel() == 0:
            continue

        x1 = (boxes[:, 0:1] - expand)
        y1 = (boxes[:, 1:2] - expand)
        x2 = (boxes[:, 2:3] + expand)
        y2 = (boxes[:, 3:4] + expand)

        inside = (cxv[None, :] >= x1) & (cxv[None, :] <= x2) & (cyv[None, :] >= y1) & (cyv[None, :] <= y2)  # [N,HW]
        pos_any = inside.any(dim=0)  # [HW]

        # GT center 기준 top-k 강제 양성
        gxc = (boxes[:, 0] + boxes[:, 2]) / 2
        gyc = (boxes[:, 1] + boxes[:, 3]) / 2
        d2 = (cxv[None, :] - gxc[:, None]) ** 2 + (cyv[None, :] - gyc[:, None]) ** 2  # [N,HW]
        force = torch.zeros_like(pos_any)
        if d2.numel():
            topk_idx = torch.topk(d2, k=min(k, d2.shape[1]), dim=1, largest=False).indices  # [N,k]
            force[topk_idx.flatten()] = True

        pos = pos_any | force
        pos_mask_flat[b] = pos
        obj_tgt[b].view(1, -1)[:, pos] = 1.0

    return obj_tgt, pos_mask_flat

# =========================
# DFL projection (inference/IoU monitoring)
# =========================
def dfl_project(reg_logits: torch.Tensor, bins: int):
    """
    reg_logits: [B, HW, 4*bins] → [B, HW, 4]
    """
    B = reg_logits.shape[0]
    x = reg_logits.view(B, -1, 4, bins)               # [B,HW,4,bins]
    prob = torch.softmax(x, dim=-1)
    idx = torch.arange(bins, device=reg_logits.device, dtype=torch.float32)
    dist = (prob * idx).sum(dim=-1)                    # [B,HW,4]
    return dist

# =========================
# Train loop
# =========================
def train_one_epoch(model, loader, device, args, csv_writer=None):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = BCEWithLogitsLossWeighted(pos_weight=1.5)
    dfl_crit = DFLLoss(bins=args.bins) if args.use_dfl else None

    total = len(loader)
    ema = 0.9
    sm_loss = None

    # 간단 선형 warmup
    global_step = 0
    base_lr = args.lr
    warmup = max(0, int(args.warmup_steps))

    pbar = tqdm(enumerate(loader), total=total, ncols=120, desc=f"Epoch {args.ep_cur}/{args.epochs}", leave=True)

    for it, (imgs, boxes_list, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)

        # warmup lr
        if warmup > 0 and global_step < warmup:
            warm_lr = base_lr * float(global_step + 1) / warmup
            for g in opt.param_groups:
                g["lr"] = warm_lr
        else:
            for g in opt.param_groups:
                g["lr"] = base_lr

        out = model(imgs)
        (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
        B, _, H, W = obj_o.shape
        s = model.stride

        # grid centers
        cx, cy = build_grid(H, W, s, imgs.device)
        cxv, cyv = cx.view(-1), cy.view(-1)  # [HW]

        # decode (for IoU monitoring only)
        if args.use_dfl:
            reg_logits = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4 * args.bins)
            reg = dfl_project(reg_logits, args.bins).clamp(min=0)
        else:
            reg = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(min=0)

        boxes_pred = dist2bbox_ltbr(reg, cxv, cyv).view(B, -1, 4).clamp(0, args.imgsz)

        # targets
        obj_tgt, pos_mask_flat = make_targets(
            boxes_list, cx, cy, B, stride=s,
            pos_expand=args.pos_expand,
            assign_topk=args.assign_topk
        )

        # DFL loss (양성 포인트에 대해 stride-정규화 거리 타깃 구축)
        dfl_loss = torch.tensor(0.0, device=device)
        if args.use_dfl and pos_mask_flat.any():
            dist_targets = torch.zeros((B, H * W, 4), device=device)
            for b in range(B):
                pos = pos_mask_flat[b]          # bool mask
                if not pos.any():
                    continue
                gts = boxes_list[b].to(device)  # [Ng,4]
                if gts.numel() == 0:
                    continue

                # 각 포인트 → 가장 가까운 GT center로 배정
                gxc = (gts[:, 0] + gts[:, 2]) / 2
                gyc = (gts[:, 1] + gts[:, 3]) / 2
                cpos = torch.stack([cxv[pos], cyv[pos]], dim=1)  # [P,2]
                cgt = torch.stack([gxc, gyc], dim=1)             # [Ng,2]
                d2 = (cpos[:, None, :] - cgt[None, :, :]).pow(2).sum(-1)  # [P,Ng]
                idx = d2.argmin(dim=1)  # [P]

                x1 = gts[idx, 0]; y1 = gts[idx, 1]; x2 = gts[idx, 2]; y2 = gts[idx, 3]
                cxp = cxv[pos]; cyp = cyv[pos]

                l = (cxp - x1) / s
                t = (cyp - y1) / s
                r = (x2 - cxp) / s
                btm = (y2 - cyp) / s
                dist_targets[b, pos] = torch.stack([l, t, r, btm], dim=1)

            # raw logits 감독
            dfl_loss = dfl_crit(reg_o, dist_targets, pos_mask_flat)

        # obj/cls BCE
        l_obj = bce(obj_o, obj_tgt)
        l_cls = bce(cls_o, obj_tgt.expand_as(cls_o))

        # IoU loss (monitoring)
        l_box = torch.tensor(0.0, device=device)
        if pos_mask_flat.any():
            ious_all = []
            for b in range(B):
                pos = pos_mask_flat[b]
                if not pos.any():
                    continue
                pred_b = boxes_pred[b][pos]
                gts = boxes_list[b].to(device)
                if gts.numel() == 0:
                    continue
                # 각 pred에 대한 GT 최고 IoU 평균(간단 모니터링)
                ious_bn = []
                for g in gts:
                    ious = bbox_iou_xyxy(pred_b, g.unsqueeze(0).expand_as(pred_b))
                    ious_bn.append(ious)
                ious = torch.stack(ious_bn, dim=1).max(dim=1).values
                ious = torch.nan_to_num(ious, nan=0.0).clamp(0, 1)
                ious_all.append(1.0 - ious.mean())
            if ious_all:
                l_box = torch.stack(ious_all).mean()

        loss = 1.0 * l_box + 1.2 * l_obj + 0.4 * l_cls + ((args.dfl_weight * dfl_loss) if args.use_dfl else 0.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        global_step += 1

        sm_loss = loss.item() if sm_loss is None else (ema * sm_loss + (1 - ema) * loss.item())
        pbar.set_postfix(
            loss=f"{sm_loss:.3f}",
            box=f"{l_box.item():.3f}",
            obj=f"{l_obj.item():.3f}",
            cls=f"{l_cls.item():.3f}",
            dfl=(f"{dfl_loss.item():.3f}" if args.use_dfl else "-"),
            lr=f"{opt.param_groups[0]['lr']:.2e}"
        )

        if csv_writer is not None and (it % 20 == 0):
            csv_writer.writerow([args.ep_cur, it + 1, total,
                                 float(loss.item()), float(l_box.item()),
                                 float(l_obj.item()), float(l_cls.item()),
                                 float(dfl_loss.item() if args.use_dfl else 0.0),
                                 float(opt.param_groups[0]['lr'])])

# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    DATA_ROOT = Path(args.data_root)
    TRAIN_IMGS = DATA_ROOT / "images/train"
    TRAIN_LBLS = DATA_ROOT / "labels/train"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Dataset / Loader
    train_ds = YOLOTxtPlateDataset(TRAIN_IMGS, TRAIN_LBLS, args.imgsz, letterbox=bool(args.letterbox))
    print(f"[data] train: {len(train_ds)} | imgsz: {args.imgsz} | letterbox={bool(args.letterbox)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2 if device == "cuda" else 0,
                              pin_memory=(device == "cuda"),
                              collate_fn=collate_fn)

    # Model
    model = LP_YOLO_Fuse(in_ch=3, num_classes=1,
                         use_psa=bool(args.use_psa),
                         use_dfl=bool(args.use_dfl),
                         bins=args.bins).to(device)

    # Output dirs
    base_out = Path(args.out_dir) if args.out_dir else DATA_ROOT.parent
    RUN_DIR = base_out / f"runs_{args.run}"
    CK_DIR = RUN_DIR / "ckpts"
    CSV_PATH = RUN_DIR / "train_log.csv"
    CK_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["epoch", "iter", "total_iter", "loss", "loss_box", "loss_obj", "loss_cls", "loss_dfl", "lr"])

        for ep in range(1, args.epochs + 1):
            args.ep_cur = ep  # 진행중인 epoch 표시용
            print(f"\n===== Epoch {ep}/{args.epochs} =====")
            train_one_epoch(model, train_loader, device, args, csv_writer=csv_writer)

            # save per-epoch
            ck = CK_DIR / f"ep{ep:03d}.pt"
            torch.save(model.state_dict(), ck)
            print(f"[ckpt] {ck}")

    # save final
    out_w = RUN_DIR / "lp_yolo_fuse_lp.pt"
    torch.save(model.state_dict(), out_w)
    print(f"[save] {out_w}")

if __name__ == "__main__":
    main()
