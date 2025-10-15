# tools/eval_recall_latency.py
# -*- coding: utf-8 -*-
"""
PSA/DFL 자동 감지 평가 스크립트
- PSA: 가중치 로드 실패 시 자동 OFF 재시도(또는 옵션으로 강제)
- DFL: state_dict에서 자동 추정(없으면 추론단에서 reg 채널 수로 자동 처리)
"""

import time, argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from src.model import LP_YOLO_Fuse
from src.utils_common import dist2bbox_ltbr


# ---------------------------
# I/O utils
# ---------------------------
def load_img(path, imgsz):
    im = Image.open(path).convert("RGB")
    im = im.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return img, im


def build_grid_xy(H, W, stride, device):
    ys = torch.arange(H, device=device) + 0.5
    xs = torch.arange(W, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    return (cx * stride).view(-1), (cy * stride).view(-1)  # [HW],[HW]


def read_gt_txt(txt_path, imgsz):
    if not txt_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32)
    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    boxes = []
    for l in lines:
        c, cx, cy, w, h = l.split()
        cx, cy, w, h = map(float, (cx, cy, w, h))
        x1 = (cx - w / 2) * imgsz
        y1 = (cy - h / 2) * imgsz
        x2 = (cx + w / 2) * imgsz
        y2 = (cy + h / 2) * imgsz
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


# ---------------------------
# DFL utils
# ---------------------------
def infer_dfl_from_ckpt(sd):
    """
    state_dict의 reg 헤드 마지막 conv out-ch으로 DFL 사용 여부/빈수 추정.
    - reg_o.1.weight 또는 reg_m.1.weight 찾음
    """
    cand = [k for k in sd.keys() if k.endswith("reg_o.1.weight") or k.endswith("reg_m.1.weight")]
    for k in cand:
        v = sd[k]
        if v.ndim == 4:
            out_ch = v.shape[0]
            if out_ch != 4 and out_ch % 4 == 0:
                return True, out_ch // 4
            else:
                return False, 8
    return False, 8


def dfl_project(reg_logits, bins: int):
    """
    reg_logits: [B, HW, 4*bins] → 기대값 기반 거리 [B, HW, 4]
    """
    B = reg_logits.shape[0]
    x = reg_logits.view(B, -1, 4, bins)               # [B, HW, 4, bins]
    prob = torch.softmax(x, dim=-1)                   # softmax over bins
    idx = torch.arange(bins, device=reg_logits.device, dtype=torch.float32)
    dist = (prob * idx).sum(dim=-1)                   # [B, HW, 4]
    return dist


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def infer_one(model, img, conf_thr=0.2, iou_thr=0.5):
    device = next(model.parameters()).device
    img = img.to(device)

    t0 = time.perf_counter()
    out = model(img)
    (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
    t1 = time.perf_counter()

    B, _, H, W = obj_o.shape
    s = model.stride
    cx, cy = build_grid_xy(H, W, s, device)  # [HW]

    # ---- DFL 자동 처리: 채널 수로 판단 ----
    if reg_o.shape[1] == 4:
        reg = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(min=0)     # [B,HW,4]
    else:
        bins = reg_o.shape[1] // 4
        reg_logits = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4 * bins)     # [B,HW,4*bins]
        reg = dfl_project(reg_logits, bins).clamp(min=0)                    # [B,HW,4]

    boxes = dist2bbox_ltbr(reg[0], cx, cy)                                  # [HW,4] (px)
    scores_obj = torch.sigmoid(obj_o[0, 0].flatten())
    scores_cls = torch.sigmoid(cls_o[0, 0].flatten())                       # 단일 클래스
    scores = scores_obj * scores_cls

    mask = scores > conf_thr
    if mask.sum() == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device), (t1 - t0) * 1000.0

    boxes = boxes[mask]
    scores = scores[mask]

    keep = nms(boxes, scores, iou_thr)
    boxes = boxes[keep]
    scores = scores[keep]

    latency_ms = (t1 - t0) * 1000.0
    return boxes, scores, latency_ms


def iou_matrix(a, b, eps=1e-9):
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    x1 = torch.max(a[:, None, 0], b[None, :, 0]); y1 = torch.max(a[:, None, 1], b[None, :, 1])
    x2 = torch.min(a[:, None, 2], b[None, :, 2]); y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter + eps)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"C:/Users/win/Desktop/lp_yolo_fuse/data")
    ap.add_argument("--weights",   default=r"C:/Users/win/Desktop/lp_yolo_fuse/lp_yolo_fuse_lp.pt")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf",  type=float, default=0.2)
    ap.add_argument("--iou",   type=float, default=0.5)
    ap.add_argument("--max_eval", type=int, default=5000, help="최대 평가 이미지 수(빠른 확인용)")

    # ▼ 자동/강제 옵션: -1=auto, 0/1=강제
    ap.add_argument("--use_psa", type=int, default=-1, help="-1:auto, 0:off, 1:on")
    ap.add_argument("--use_dfl", type=int, default=-1, help="-1:auto, 0:off, 1:on")
    ap.add_argument("--bins",    type=int, default=8)
    args = ap.parse_args()

    root   = Path(args.data_root)
    imgdir = root / "images/val"
    lbldir = root / "labels/val"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 가중치 로드 ----
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    sd   = ckpt if isinstance(ckpt, dict) else ckpt

    # ---- DFL 자동 추정 또는 강제 ----
    if args.use_dfl == -1:
        use_dfl, bins = infer_dfl_from_ckpt(sd)
    else:
        use_dfl = bool(args.use_dfl); bins = int(args.bins)

    # ---- PSA 자동/강제: 먼저 ON 시도, 실패 시 OFF 재시도 ----
    psa_pref = args.use_psa  # -1/0/1

    def build(psa_flag: bool):
        return LP_YOLO_Fuse(in_ch=3, num_classes=1, use_psa=psa_flag, use_dfl=use_dfl, bins=bins).to(device)

    psa_try_on = (True if psa_pref == -1 else bool(psa_pref))
    model = build(psa_try_on)
    loaded_with_psa = psa_try_on
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        if ("neck.psa" in str(e)) and (psa_pref != 1):
            model = build(False)
            model.load_state_dict(sd, strict=False)
            loaded_with_psa = False
        else:
            model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"[model] device={device} | stride={getattr(model, 'stride', 'N/A')} | PSA={loaded_with_psa} | DFL={use_dfl} (bins={bins})")

    img_paths = sorted([p for p in imgdir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if args.max_eval and len(img_paths) > args.max_eval:
        img_paths = img_paths[:args.max_eval]

    total_gt = 0
    total_tp = 0
    lat_list = []

    for i, ip in enumerate(img_paths, 1):
        lp = lbldir / f"{ip.stem}.txt"
        gt = read_gt_txt(lp, args.imgsz).to(device)   # [Ng,4]

        img, _ = load_img(ip, args.imgsz)
        boxes, scores, lat_ms = infer_one(model, img, args.conf, args.iou)

        lat_list.append(lat_ms)
        total_gt += gt.shape[0]

        if gt.numel() and boxes.numel():
            ious = iou_matrix(boxes, gt)             # [Np,Ng]
            max_iou_per_gt = ious.max(dim=0).values
            total_tp += int((max_iou_per_gt >= 0.5).sum().item())

        if i % 200 == 0:
            rec = (total_tp/total_gt) if total_gt>0 else 0.0
            print(f"[eval] {i}/{len(img_paths)}  TP/GT={total_tp}/{total_gt}  Recall={rec:.4f}  Lat(ms)~{np.mean(lat_list):.1f}")

    recall = (total_tp/total_gt) if total_gt>0 else 0.0
    avg_lat = float(np.mean(lat_list)) if lat_list else 0.0

    print("\n===== EVAL RESULT =====")
    print(f"Images      : {len(img_paths)}")
    print(f"GT Boxes    : {total_gt}")
    print(f"True Pos    : {total_tp}")
    print(f"Recall@0.5  : {recall:.4f}")
    print(f"Avg Latency : {avg_lat:.1f} ms  (batch=1)")

if __name__ == "__main__":
    main()
