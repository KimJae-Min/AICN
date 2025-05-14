#!/usr/bin/env python3
import json
import random
import csv
import re
import unicodedata
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ───── 설정 ─────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
RAW_FULL_ANN_DIR   = PROJECT_ROOT / "raw/full_vehicle/annotations"
RAW_FULL_IMG_ROOT  = PROJECT_ROOT / "raw/full_vehicle/images"
RAW_CROP_DIR       = PROJECT_ROOT / "raw/cropped_plate"
OUT_DET            = PROJECT_ROOT / "datasets/detection"
OUT_REC            = PROJECT_ROOT / "datasets/recognition"
DATA_DIR           = PROJECT_ROOT / "data"
VAL_RATIO          = 0.2
CLASS_ID           = 0
CLASS_NAMES        = ["license_plate"]

# ───── 1) 이미지 인덱스 구축 ─────────────────────────────────────
print("⏳ Indexing images under", RAW_FULL_IMG_ROOT)

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)

IMG_IDX = {
    norm(p.name): p
    for p in RAW_FULL_IMG_ROOT.rglob("*")
    if p.is_file()
}
print(f"📂 Found {len(IMG_IDX)} images")

def find_img(filename: str) -> Path:
    fn = norm(filename)
    # 1) exact
    if fn in IMG_IDX:
        return IMG_IDX[fn]
    # 2) suffix
    for name, path in IMG_IDX.items():
        if name.endswith(fn):
            return path
    # 3) substring
    for name, path in IMG_IDX.items():
        if fn in name:
            return path
    # 4) 접두사 제거
    m = re.match(r'^\d+_\d+_(.+)$', fn)
    if m:
        rest = m.group(1)
        for name, path in IMG_IDX.items():
            if name.endswith(rest):
                return path
    raise FileNotFoundError(f"{filename} not found")

# ───── 2) 전체 어노테이션 로드 & 셔플 ────────────────────────────
all_ann = []
print("⏳ Loading annotation entries…")
for jf in tqdm(list(RAW_FULL_ANN_DIR.rglob("*.json")), desc="JSON files"):
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(data, list):
        all_ann.extend(data)
    elif isinstance(data, dict) and "annotations" in data:
        all_ann.extend(data["annotations"])
    else:
        all_ann.append(data)
print(f"▶ Total annotation entries: {len(all_ann):,}")

random.shuffle(all_ann)
n_val = int(len(all_ann) * VAL_RATIO)
train_entries, val_entries = all_ann[n_val:], all_ann[:n_val]
print(f"▶ Train/Val split: {len(train_entries):,}/{len(val_entries):,}")

# ───── 3) Detection 전처리 ───────────────────────────────────
# images/ 와 labels/ 폴더 구조 미리 생성
(OUT_DET / "images" / "train").mkdir(parents=True, exist_ok=True)
(OUT_DET / "images" / "val"  ).mkdir(parents=True, exist_ok=True)
(OUT_DET / "labels" / "train").mkdir(parents=True, exist_ok=True)
(OUT_DET / "labels" / "val"  ).mkdir(parents=True, exist_ok=True)

def yolo_line(bbox, w, h, cid=0):
    (x1, y1), (x2, y2) = bbox
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def make_detection(entries, subset):
    seen, mismatches = set(), []
    for e in tqdm(entries, desc=f"Detect→{subset}", mininterval=1.0):
        fn = Path(e["imagePath"]).name
        if fn in seen:
            continue
        try:
            src = find_img(fn)
        except FileNotFoundError:
            if len(mismatches) < 10:
                mismatches.append(fn)
            continue
        seen.add(fn)

        bbox = e.get("plate", {}).get("bbox")
        if not bbox:
            continue

        # 이미지 크기 얻기 (깨진 파일은 무시)
        try:
            w, h = Image.open(src).size
        except UnidentifiedImageError:
            print(f"⚠️ Skipping unreadable image: {src}")
            continue

        # 1) 심볼릭 링크 생성 (실제 복사 없이)
        dst_img = OUT_DET/"images"/subset/src.name
        if not dst_img.exists():
            dst_img.symlink_to(src)

        # 2) 라벨(.txt) 생성
        line = yolo_line(bbox, w, h, CLASS_ID)
        lbl = OUT_DET/"labels"/subset/f"{dst_img.stem}.txt"
        lbl.write_text(line + "\n", encoding="utf-8")

    print(f"▶ sample mismatches: {mismatches}")
    print(f"✅ {subset} detection done: {len(seen):,} images")

make_detection(train_entries, "train")
make_detection(val_entries,   "val")

# custom_det.yaml 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
yaml = DATA_DIR / "custom_det.yaml"
yaml.write_text(
    f"train: { (OUT_DET/'images'/'train').resolve() }\n"
    f"val:   { (OUT_DET/'images'/'val').resolve() }\n\n"
    f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
)
print("✅ Detection config written to", yaml)

# ───── 4) Recognition 전처리 ───────────────────────────────────
for subset in ("train", "val"):
    label_dir = RAW_CROP_DIR / subset / "labels"
    img_root  = RAW_CROP_DIR / subset / "images"
    out_img   = OUT_REC / "images" / subset
    out_csv   = OUT_REC / "labels" / f"{subset}.csv"

    out_img.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [["image", "plate"]]
    for jf in label_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ JSON 로드 실패: {jf} ({e})")
            continue

        img_name  = data.get("imagePath")
        plate_txt = data.get("value", "")
        if not img_name:
            continue

        src_crop = img_root / img_name
        if not src_crop.exists():
            print(f"⚠️ 크롭 이미지 누락: {src_crop} — 건너뜁니다.")
            continue

        dst_crop = out_img / img_name
        if not dst_crop.exists():
            dst_crop.symlink_to(src_crop)

        rows.append([str(dst_crop.resolve()), plate_txt])

    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerows(rows)
    print(f"✅ {subset} recognition CSV: {len(rows)-1:,} entries")

print("🎉 Preprocessing complete.")
