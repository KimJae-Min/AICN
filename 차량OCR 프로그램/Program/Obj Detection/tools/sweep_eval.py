# tools/sweep_eval.py
# -*- coding: utf-8 -*-
import argparse, time, math, csv, itertools
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

# 프로젝트 모듈
from src.model import LP_YOLO_Fuse
from src.utils_common import dist2bbox_ltbr

# =========================
# Helpers
# =========================
def parse_list_or_range(s: str, cast=float):
    """
    "0.20,0.25,0.30" -> [0.20,0.25,0.30]
    "0.20:0.05:0.50" -> [0.20,0.25,...,0.50]
    """
    s = str(s).strip()
    if ":" in s:
        a, step, b = s.split(":")
        a, step, b = cast(a), cast(step), cast(b)
        out = []
        x = a
        # 범위 포함
        # 부동소수 오차 방지용 작은 이득
        while (x <= b + 1e-9) if step > 0 else (x >= b - 1e-9):
            out.append(cast(round(x, 6)))
            x += step
        return out
    else:
        return [cast(x) for x in s.split(",") if x != ""]

def load_img(path, imgsz):
    im = Image.open(path).convert("RGB")
    im = im.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return img, im  # tensor, PIL

def read_gt_txt(txt_path, imgsz):
    if not txt_path.exists():
        return torch.zeros((0,4), dtype=torch.float32)
    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    boxes=[]
    for l in lines:
        parts = l.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        cx, cy, w, h = map(float, (cx,cy,w,h))
        x1 = (cx - w/2) * imgsz
        y1 = (cy - h/2) * imgsz
        x2 = (cx + w/2) * imgsz
        y2 = (cy + h/2) * imgsz
        boxes.append([x1,y1,x2,y2])
    if not boxes:
        return torch.zeros((0,4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)

def build_grid_xy(H, W, stride, device):
    ys = torch.arange(H, device=device) + 0.5
    xs = torch.arange(W, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    return (cx*stride).view(-1), (cy*stride).view(-1)  # [HW],[HW]

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

def iou_matrix(a, b, eps=1e-9):
    # a: [Na,4], b:[Nb,4]
    if a.numel()==0 or b.numel()==0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    x1 = torch.max(a[:,None,0], b[None,:,0])
    y1 = torch.max(a[:,None,1], b[None,:,1])
    x2 = torch.min(a[:,None,2], b[None,:,2])
    y2 = torch.min(a[:,None,3], b[None,:,3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    aa = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
    bb = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
    return inter / (aa[:,None] + bb[None,:] - inter + eps)

# =========================
# CKPT → 구조 자동 감지
# =========================
def inspect_ckpt_config(ckpt_state: dict):
    """
    state_dict를 보고 PSA/DFL/bins 추정
    - PSA: 키에 'neck.psa.' 포함 여부
    - DFL: head.reg_o.1.weight out_channels == 4*bins (4이면 DFL off)
    """
    keys = list(ckpt_state.keys())
    has_psa = any(k.startswith("neck.psa.") for k in keys)

    # reg head 채널 파악
    # 일반적으로 conv의 bias 키 사용
    reg_bias_keys = [k for k in keys if "head.reg_o.1.bias" in k]
    bins = 1
    use_dfl = False
    if reg_bias_keys:
        out_ch = int(ckpt_state[reg_bias_keys[0]].numel())
        if out_ch % 4 == 0:
            bins = out_ch // 4
            use_dfl = (bins > 1)
    else:
        # fallback: weight에서 out_channels
        reg_w_keys = [k for k in keys if "head.reg_o.1.weight" in k]
        if reg_w_keys:
            out_ch = int(ckpt_state[reg_w_keys[0]].shape[0])
            if out_ch % 4 == 0:
                bins = out_ch // 4
                use_dfl = (bins > 1)

    return has_psa, use_dfl, bins

def build_model_for_ckpt(ckpt_state, device, force_psa=None, force_dfl=None, force_bins=None):
    ck_psa, ck_dfl, ck_bins = inspect_ckpt_config(ckpt_state)
    if force_psa is not None: ck_psa = bool(force_psa)
    if force_dfl is not None: ck_dfl = bool(force_dfl)
    if force_bins is not None: ck_bins = int(force_bins)

    model = LP_YOLO_Fuse(in_ch=3, num_classes=1,
                         use_psa=ck_psa, use_dfl=ck_dfl, bins=ck_bins).to(device)
    print(f"[ckpt-config] PSA={ck_psa} | DFL={ck_dfl} | bins={ck_bins}")
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        # PSA 미사용 ckpt를 PSA ON 모델에 로드한 경우 등, 정보성 출력
        miss_short = [m for m in missing if ("running_" not in m)]
        if miss_short and any("neck.psa." in m for m in miss_short):
            print("[hint] ckpt에 PSA 레이어 파라미터가 없습니다. PSA=False 모델로 학습된 가중치로 보입니다.")
    model.eval()
    return model

# =========================
# Inference (DFL 자동)
# =========================
@torch.no_grad()
def infer_one(model, img, conf_thr=0.2, iou_thr=0.5, keep_topk=1):
    device = next(model.parameters()).device
    img = img.to(device)
    t0 = time.perf_counter()
    out = model(img)
    (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
    t1 = time.perf_counter()

    B, _, H, W = obj_o.shape
    s = model.stride
    cx, cy = build_grid_xy(H, W, s, device)  # [HW]

    # DFL 자동 처리
    ch = reg_o.shape[1]
    if ch % 4 == 0 and ch > 4:
        bins = ch // 4
        reg_logits = reg_o.permute(0,2,3,1).reshape(B, -1, 4*bins)
        reg = dfl_project(reg_logits, bins).clamp(min=0)
    else:
        reg = reg_o.permute(0,2,3,1).reshape(B,-1,4).clamp(min=0)

    boxes = dist2bbox_ltbr(reg[0], cx, cy)                      # [HW,4] (px)
    scores_obj = torch.sigmoid(obj_o[0,0].flatten())            # [HW]
    scores_cls = torch.sigmoid(cls_o[0,0].flatten())            # 단일 클래스
    scores = scores_obj * scores_cls

    # threshold → topk → NMS
    mask = scores > conf_thr
    if mask.sum() == 0:
        return (torch.zeros((0,4), device=device),
                torch.zeros((0,), device=device),
                (t1-t0)*1000.0)

    boxes = boxes[mask]; scores = scores[mask]
    if keep_topk is not None and keep_topk > 0 and boxes.shape[0] > keep_topk:
        topk = torch.topk(scores, k=keep_topk)
        boxes = boxes[topk.indices]
        scores = topk.values

    keep = nms(boxes, scores, iou_thr)
    boxes = boxes[keep]
    scores = scores[keep]
    latency_ms = (t1 - t0) * 1000.0
    return boxes, scores, latency_ms

# =========================
# Metrics
# =========================
def eval_dataset(model, img_paths, lbl_dir, imgsz, conf, nms_iou, keep_topk, iou_match=0.5, log_every=200):
    device = next(model.parameters()).device
    total_gt = 0
    TP = 0
    FP = 0
    lat_list = []

    for i, ip in enumerate(img_paths, 1):
        lp = lbl_dir / f"{ip.stem}.txt"
        gt = read_gt_txt(lp, imgsz).to(device)   # [Ng,4]

        img, _ = load_img(ip, imgsz)             # [1,3,H,W]
        boxes, scores, lat_ms = infer_one(model, img, conf_thr=conf, iou_thr=nms_iou, keep_topk=keep_topk)
        lat_list.append(lat_ms)

        total_gt += gt.shape[0]

        if boxes.numel() == 0:
            # 예측 없음
            continue

        if gt.numel() == 0:
            # GT 없음 → 전부 FP
            FP += boxes.shape[0]
            continue

        # GT-예측 최대 매칭 (1:1 보장은 안 하지만 대략적 성능 비교에는 충분)
        ious = iou_matrix(boxes, gt)  # [Np,Ng]
        max_iou_per_pred, gt_idx = ious.max(dim=1)
        # 한 GT가 여러 번 카운트되는 것을 막기 위해 간단한 1:1 제약(탐욕) 추가
        used_gt = set()
        tp_local = 0
        for k, iou_v in enumerate(max_iou_per_pred.tolist()):
            gid = int(gt_idx[k].item())
            if iou_v >= iou_match and gid not in used_gt:
                tp_local += 1
                used_gt.add(gid)
        fp_local = boxes.shape[0] - tp_local
        TP += tp_local
        FP += fp_local

        if (i % log_every) == 0:
            R = (TP / total_gt) if total_gt else 0.0
            P = TP / max(1, (TP + FP))
            print(f"[eval] {i}/{len(img_paths)}  P={P:.3f} R={R:.3f}  Lat~{np.mean(lat_list):.1f}ms")

    FN = total_gt - TP
    P = TP / max(1, (TP + FP))
    R = TP / max(1, total_gt)
    F1 = (2*P*R / max(1e-9, (P+R))) if (P+R) > 0 else 0.0
    lat = float(np.mean(lat_list)) if lat_list else 0.0

    return dict(
        images=len(img_paths),
        gt=total_gt,
        tp=TP, fp=FP, fn=FN,
        precision=P, recall=R, f1=F1,
        latency_ms=lat
    )

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"C:/Users/win/Desktop/lp_yolo_fuse/data")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--iou",  type=float, default=0.5, help="GT 매칭 IoU")
    ap.add_argument("--max_eval", type=int, default=20000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # 스윕 대상
    ap.add_argument("--confs", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--nms_ious", default="0.30,0.45")
    ap.add_argument("--keep_topks", default="1,2")

    # 강제 구성 (옵션)
    ap.add_argument("--force_psa", type=int, default=None, help="0/1 (없으면 ckpt 자동감지)")
    ap.add_argument("--force_dfl", type=int, default=None, help="0/1 (없으면 ckpt 자동감지)")
    ap.add_argument("--force_bins", type=int, default=None, help="DFL bins 강제")

    # 결과 저장/요약
    ap.add_argument("--run", default="sweep")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--xlsx", type=int, default=1, help="1: pandas로 엑셀 저장 시도(없으면 CSV만)")
    ap.add_argument("--notes", default="", help="결과 파일에 메모 기록")
    ap.add_argument("--sort_by", default="f1", choices=["f1","precision","recall","latency"])
    ap.add_argument("--topk", type=int, default=10, help="정렬 후 상위 N 출력")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    device = torch.device(args.device)
    root = Path(args.data_root)
    imgdir = root / "images" / "val"
    lbldir = root / "labels" / "val"
    img_paths = sorted([p for p in imgdir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
    if args.max_eval and len(img_paths) > args.max_eval:
        img_paths = img_paths[:args.max_eval]

    # 출력 경로
    base_out = Path(args.out_dir) if args.out_dir else (root.parent / f"runs_{args.run}")
    base_out.mkdir(parents=True, exist_ok=True)
    csv_path = base_out / "sweep_results.csv"
    xlsx_path = base_out / "sweep_results.xlsx"

    # 체크포인트 로드 & 모델 구성 자동 감지
    print(f"[load ckpt] {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ck = ckpt["state_dict"]
    else:
        ck = ckpt
    model = build_model_for_ckpt(ck, device,
                                 force_psa=args.force_psa,
                                 force_dfl=args.force_dfl,
                                 force_bins=args.force_bins)

    # 스윕 리스트 파싱
    conf_list = parse_list_or_range(args.confs, float)
    nms_list  = parse_list_or_range(args.nms_ious, float)
    topk_list = parse_list_or_range(args.keep_topks, int)

    # 헤더 쓰기
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "weights","notes","imgsz","iou_match","images","gt",
            "conf","nms_iou","keep_topk",
            "tp","fp","fn","precision","recall","f1","latency_ms"
        ])

        results = []
        total_cases = len(conf_list) * len(nms_list) * len(topk_list)
        case_id = 0

        for conf, nms_iou, keep_topk in itertools.product(conf_list, nms_list, topk_list):
            case_id += 1
            print(f"\n=== Case {case_id}/{total_cases} :: conf={conf} | nms_iou={nms_iou} | keep_topk={keep_topk} ===")
            metrics = eval_dataset(model, img_paths, lbldir, args.imgsz,
                                   conf=conf, nms_iou=nms_iou, keep_topk=keep_topk,
                                   iou_match=args.iou, log_every=args.log_every)

            print("===== EVAL RESULT =====")
            print(f"Images      : {metrics['images']}")
            print(f"GT Boxes    : {metrics['gt']}")
            print(f"TP / FP / FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
            print(f"Precision   : {metrics['precision']:.4f}")
            print(f"Recall@{args.iou:.2f}: {metrics['recall']:.4f}")
            print(f"F1          : {metrics['f1']:.4f}")
            print(f"Avg Latency : {metrics['latency_ms']:.1f} ms  (batch=1)")

            row = [
                str(args.weights), args.notes, args.imgsz, args.iou, metrics['images'], metrics['gt'],
                conf, nms_iou, keep_topk,
                metrics['tp'], metrics['fp'], metrics['fn'],
                round(metrics['precision'], 6), round(metrics['recall'], 6),
                round(metrics['f1'], 6), round(metrics['latency_ms'], 3)
            ]
            wr.writerow(row)

            rec = dict(
                weights=str(args.weights),
                notes=args.notes,
                imgsz=args.imgsz,
                iou_match=args.iou,
                images=metrics['images'], gt=metrics['gt'],
                conf=conf, nms_iou=nms_iou, keep_topk=keep_topk,
                tp=metrics['tp'], fp=metrics['fp'], fn=metrics['fn'],
                precision=metrics['precision'], recall=metrics['recall'],
                f1=metrics['f1'], latency_ms=metrics['latency_ms']
            )
            results.append(rec)

    # 정렬/상위 출력 & 엑셀 저장(옵션)
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        key = {"f1":"f1","precision":"precision","recall":"recall","latency":"latency_ms"}[args.sort_by]
        asc = (args.sort_by == "latency")
        df_sorted = df.sort_values(key, ascending=asc)
        top_df = df_sorted.head(args.topk)
        print("\n=== TOP results ===")
        print(top_df.to_string(index=False))
        if int(args.xlsx) == 1:
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
                df_sorted.to_excel(w, index=False, sheet_name="sweep")
                top_df.to_excel(w, index=False, sheet_name="top")
            print(f"[saved] {xlsx_path}")
        print(f"[saved] {csv_path}")
    except Exception as e:
        print(f"[pandas] 건너뜀({e}). CSV만 저장됨: {csv_path}")

if __name__ == "__main__":
    main()
