# -*- coding: utf-8 -*-
# tools/labels2yolo_plate_agg.py
import os, json
from pathlib import Path
from PIL import Image

def gather_jsons(labels_root):
    files = list(Path(labels_root).rglob("*.json"))
    return files

def parse_bbox_from_json(jpath):
    js = json.load(open(jpath, encoding="utf-8"))
    img_name = js.get("imagePath") or ""
    plate = js.get("plate")
    bbs=[]
    # plate.bbox = [[x1,y1],[x2,y2]]
    if isinstance(plate, dict) and isinstance(plate.get("bbox"), list) and len(plate["bbox"])==2:
        (x1,y1),(x2,y2) = plate["bbox"]
        x,y,w,h = float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1)
        if w>0 and h>0: bbs.append((x,y,w,h))
    return img_name, bbs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--labels_root", required=True)
    ap.add_argument("--out_labels", required=True)     # 예: data/labels/train
    ap.add_argument("--gather_images_to", default=None) # 예: data/images/train
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    img_index = {}
    for ext in (".jpg",".jpeg",".png",".bmp"):
        for p in Path(args.images_root).rglob(f"*{ext}"):
            img_index[p.name] = str(p)

    groups = {}  # imageName -> list of (x,y,w,h)
    for jp in gather_jsons(args.labels_root):
        try:
            img_name, bbs = parse_bbox_from_json(jp)
            if not img_name: continue
            key = os.path.basename(img_name)
            if key not in groups: groups[key]=[]
            groups[key].extend(bbs)
        except Exception:
            continue

    outL = Path(args.out_labels); outL.mkdir(parents=True, exist_ok=True)
    outI = None
    if args.gather_images_to:
        outI = Path(args.gather_images_to); outI.mkdir(parents=True, exist_ok=True)

    n_ok=n_skip=0
    for img_name, bbs in groups.items():
        full = img_index.get(img_name)
        if not full:
            # 일부 json은 현재 남은 이미지에 없을 수 있음
            n_skip += 1; continue
        try:
            with Image.open(full) as im: W,H = im.size
        except Exception: n_skip += 1; continue

        lines=[]
        for (x,y,w,h) in bbs:
            if w<=0 or h<=0: continue
            xc=(x+w/2)/W; yc=(y+h/2)/H; nw=w/W; nh=h/H
            xc=max(0,min(1,xc)); yc=max(0,min(1,yc))
            nw=max(0,min(1,nw)); nh=max(0,min(1,nh))
            if nw==0 or nh==0: continue
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(img_name).stem
        (outL/f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if outI:
            dst = outI / os.path.basename(full)
            if dst.exists(): pass
            else:
                if args.copy or os.name=="nt":
                    import shutil; shutil.copy2(full, dst)
                else:
                    os.symlink(os.path.abspath(full), dst)
        n_ok += 1

    print(f"[DONE] txt 생성: {n_ok}  |  스킵: {n_skip}  |  out={outL}")
    if outI: print(f"[OUT] images={outI}")

if __name__ == "__main__":
    main()
