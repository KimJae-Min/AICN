# -*- coding: utf-8 -*-
import argparse, time, csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from src.model import LP_YOLO_Fuse
from src.utils_common import dist2bbox_ltbr


# ------------------------
# IO / utils
# ------------------------
def load_img(path, imgsz):
    im = Image.open(path).convert("RGB")
    im = im.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return img, im  # tensor, PIL(resized)

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
        parts = l.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        x1 = (cx - w/2) * imgsz
        y1 = (cy - h/2) * imgsz
        x2 = (cx + w/2) * imgsz
        y2 = (cy + h/2) * imgsz
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)

@torch.no_grad()
def iou_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9):
    device = a.device
    b = b.to(device, non_blocking=True)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=device)

    x1 = torch.max(a[:, None, 0], b[None, :, 0])
    y1 = torch.max(a[:, None, 1], b[None, :, 1])
    x2 = torch.min(a[:, None, 2], b[None, :, 2])
    y2 = torch.min(a[:, None, 3], b[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter + eps)


# ------------------------
# PSA/DFL 자동감지
# ------------------------
def sniff_ckpt_cfg(sd):
    # DFL bins 추론 (out_ch/4)
    bins = None
    for key in ("head.reg_o.1.bias", "head.reg_m.1.bias"):
        if key in sd and isinstance(sd[key], torch.Tensor):
            out_ch = int(sd[key].numel())
            if out_ch % 4 == 0:
                bins = out_ch // 4
                break
    has_psa = ("neck.psa.conv.weight" in sd) or ("neck.psa.conv.bias" in sd)
    use_dfl = (bins is not None and bins > 1)
    if bins is None:
        bins = 1
    return has_psa, use_dfl, bins


# ------------------------
# 추론 1장 (DFL/Plain 자동 대응)
# ------------------------
@torch.no_grad()
def infer_one(model, img, conf_thr=0.2, iou_match=0.5, nms_iou=0.5, keep_topk=None, score_mode="objcls"):
    device = next(model.parameters()).device
    img = img.to(device, non_blocking=True)

    t0 = time.perf_counter()
    out = model(img)
    (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
    t1 = time.perf_counter()

    B, _, H, W = obj_o.shape
    s = model.stride
    cx, cy = build_grid_xy(H, W, s, device)  # [HW]

    # DFL / LTRB decode
    if getattr(model, "use_dfl", False) and int(getattr(model, "bins", 1)) > 1:
        bins = int(model.bins)
        reg_logits = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4 * bins)  # [B,HW,4*bins]
        x = reg_logits.view(B, -1, 4, bins)
        prob = torch.softmax(x, dim=-1)
        idx = torch.arange(bins, device=device, dtype=torch.float32)
        reg = (prob * idx).sum(dim=-1).clamp(min=0)  # [B,HW,4]
    else:
        reg = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(min=0)  # [B,HW,4]

    boxes = dist2bbox_ltbr(reg[0], cx, cy)  # [HW,4] (px)

    # scores
    scores_obj = torch.sigmoid(obj_o[0, 0].flatten())        # [HW]
    if score_mode == "obj":
        scores = scores_obj
    else:
        scores_cls = torch.sigmoid(cls_o[0, 0].flatten())    # 1-class
        scores = scores_obj * scores_cls

    # thresholding
    mask = scores > conf_thr
    if mask.sum() == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device), (t1 - t0) * 1000.0

    boxes = boxes[mask]
    scores = scores[mask]

    # NMS
    keep = nms(boxes, scores, nms_iou)
    if keep_topk is not None and keep_topk > 0:
        keep = keep[:keep_topk]
    boxes = boxes[keep]
    scores = scores[keep]

    latency_ms = (t1 - t0) * 1000.0
    return boxes, scores, latency_ms


# ------------------------
# 케이스 평가
# ------------------------
@torch.no_grad()
def evaluate_case(model, img_paths, lbldir, imgsz, conf, nms_iou, keep_topk, iou_match, score_mode):
    device = next(model.parameters()).device
    TP = 0; FP = 0; FN = 0
    lat_list = []
    total_gt = 0

    for i, ip in enumerate(img_paths, 1):
        lp = lbldir / f"{ip.stem}.txt"
        gt = read_gt_txt(lp, imgsz)     # [Ng,4] (cpu)
        img, _ = load_img(ip, imgsz)    # [1,3,H,W] (cpu)

        boxes, scores, lat_ms = infer_one(model, img, conf, iou_match, nms_iou, keep_topk, score_mode)
        lat_list.append(lat_ms)

        Ng = gt.shape[0]
        Np = boxes.shape[0]
        total_gt += Ng

        if Np == 0 and Ng == 0:
            pass
        elif Np == 0 and Ng > 0:
            FN += Ng
        elif Np > 0 and Ng == 0:
            FP += Np
        else:
            imat = iou_matrix(boxes, gt)  # [Np,Ng] (자동 device 맞춤)
            max_iou_pred = imat.max(dim=1).values
            max_iou_gt   = imat.max(dim=0).values

            tp_here = int((max_iou_gt >= iou_match).sum().item())
            fp_here = int((max_iou_pred <  iou_match).sum().item())
            fn_here = Ng - tp_here

            TP += tp_here; FP += fp_here; FN += fn_here

        if i % 200 == 0:
            prec = (TP/(TP+FP)) if (TP+FP)>0 else 0.0
            rec  = (TP/total_gt) if total_gt>0 else 0.0
            print(f"[eval] {i}/{len(img_paths)}  P={prec:.3f} R={rec:.3f}  Lat~{np.mean(lat_list):.1f}ms")

    precision = (TP/(TP+FP)) if (TP+FP)>0 else 0.0
    recall = (TP/total_gt) if total_gt>0 else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
    avg_lat = float(np.mean(lat_list)) if lat_list else 0.0

    return dict(TP=TP, FP=FP, FN=FN, GT=total_gt,
                precision=precision, recall=recall, f1=f1, avg_lat=avg_lat)


# ------------------------
# 파라미터 파서(범위/리스트)
# ------------------------
def parse_list_arg(s: str, cast=float):
    """
    "0.20:0.05:0.50"  -> [0.20,0.25,...,0.50]
    "0.30,0.45"       -> [0.30,0.45]
    "1,2,3"           -> [1,2,3]
    """
    s = s.strip()
    if ":" in s:
        a, step, b = s.split(":")
        a = cast(a); step = cast(step); b = cast(b)
        vals = []
        cur = a
        # 부동소수 오차 방지 살짝 여유
        while cur <= b + (1e-9 if cast is float else 0):
            vals.append(cast(round(cur, 10)) if cast is float else cast(cur))
            cur += step
        return vals
    else:
        return [cast(x) for x in s.split(",") if x.strip()]


# ------------------------
# 메인
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"C:/Users/win/Desktop/lp_yolo_fuse/data")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--max_eval", type=int, default=20000)
    ap.add_argument("--iou", type=float, default=0.5, help="matching IoU for metrics")

    # Sweep spaces
    ap.add_argument("--confs", default="0.20:0.05:0.50", help="ex) '0.20:0.05:0.50' or '0.20,0.25'")
    ap.add_argument("--nms_ious", default="0.30", help="ex) '0.30,0.45'")
    ap.add_argument("--keep_topks", default="1,2", help="ex) '1,2' or '0'")

    # Score mode
    ap.add_argument("--score_mode", choices=["obj","objcls"], default="objcls")

    # 강제 오버라이드(없으면 ckpt에서 자동감지)
    ap.add_argument("--force_psa", type=int, default=-1)
    ap.add_argument("--force_dfl", type=int, default=-1)
    ap.add_argument("--force_bins", type=int, default=-1)

    # Output
    ap.add_argument("--run", default="sweep")
    ap.add_argument("--out_dir", default="")           # 기본: DATA_ROOT.parent
    ap.add_argument("--xlsx", type=int, default=0)
    ap.add_argument("--notes", default="")

    return ap.parse_args()


def main():
    args = parse_args()

    # Output dirs
    DATA_ROOT = Path(args.data_root)
    base_out = Path(args.out_dir) if args.out_dir else DATA_ROOT.parent
    RUN_DIR = base_out / f"runs_{args.run}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = RUN_DIR / "sweep_results.csv"
    XLSX_PATH = RUN_DIR / "sweep_results.xlsx"

    # Data
    imgdir = DATA_ROOT / "images/val"
    lbldir = DATA_ROOT / "labels/val"
    img_paths = sorted([p for p in imgdir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if args.max_eval and len(img_paths) > args.max_eval:
        img_paths = img_paths[:args.max_eval]

    # Load weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    has_psa, use_dfl_auto, bins_auto = sniff_ckpt_cfg(sd)
    use_psa = bool(args.force_psa) if args.force_psa in (0, 1) else has_psa
    use_dfl = bool(args.force_dfl) if args.force_dfl in (0, 1) else use_dfl_auto
    bins = int(args.force_bins) if args.force_bins > 0 else int(bins_auto)
    print(f"[ckpt-config] PSA={use_psa} | DFL={use_dfl} | bins={bins}")

    model = LP_YOLO_Fuse(in_ch=3, num_classes=1, use_psa=use_psa, use_dfl=use_dfl, bins=bins).to(device)
    missing = model.load_state_dict(sd, strict=False)
    if isinstance(missing, tuple):
        miss, unexp = missing
        print(f"[load] missing={len(miss)}, unexpected={len(unexp)}")
    else:
        print("[load] state_dict loaded")
    model.eval()

    # Sweep space
    conf_list = parse_list_arg(args.confs, float)
    nms_list = parse_list_arg(args.nms_ious, float)
    topk_list = parse_list_arg(args.keep_topks, int)

    total_cases = len(conf_list) * len(nms_list) * len(topk_list)
    print(f"[sweep] cases = {len(conf_list)} x {len(nms_list)} x {len(topk_list)} = {total_cases}")

    # CSV header
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["run", "notes", "imgsz", "iou_match",
                    "conf", "nms_iou", "keep_topk",
                    "TP", "FP", "FN", "GT",
                    "precision", "recall", "f1", "avg_lat_ms"])

        case_id = 0
        for conf in conf_list:
            for nms_iou in nms_list:
                for keep_topk in topk_list:
                    case_id += 1
                    print(f"\n=== Case {case_id}/{total_cases} :: conf={conf} | nms_iou={nms_iou} | keep_topk={keep_topk} ===")

                    metrics = evaluate_case(
                        model, img_paths, lbldir,
                        imgsz=args.imgsz,
                        conf=conf, nms_iou=nms_iou, keep_topk=(None if keep_topk == 0 else keep_topk),
                        iou_match=args.iou, score_mode=args.score_mode
                    )
                    w.writerow([args.run, args.notes, args.imgsz, args.iou,
                                conf, nms_iou, keep_topk,
                                metrics["TP"], metrics["FP"], metrics["FN"], metrics["GT"],
                                f'{metrics["precision"]:.6f}', f'{metrics["recall"]:.6f}', f'{metrics["f1"]:.6f}',
                                f'{metrics["avg_lat"]:.3f}'])

    print(f"\n[save] CSV: {CSV_PATH}")

    # Optional Excel
    if int(args.xlsx) == 1:
        try:
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
            with pd.ExcelWriter(XLSX_PATH, engine="xlsxwriter") as xw:
                df.to_excel(xw, index=False, sheet_name="results")
            print(f"[save] XLSX: {XLSX_PATH}")
        except Exception as e:
            print(f"[pandas] 건너뜀({e}). CSV만 저장됨: {CSV_PATH}")

if __name__ == "__main__":
    main()
