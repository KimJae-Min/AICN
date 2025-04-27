#!/usr/bin/env python3
import json, random, csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ───── 설정 ─────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
RAW_FULL_ANN_DIR   = PROJECT_ROOT/"raw/full_vehicle/annotations"
RAW_FULL_IMG_ROOT  = PROJECT_ROOT/"raw/full_vehicle/images"
RAW_CROP_DIR       = PROJECT_ROOT/"raw/cropped_plate"
OUT_DET            = PROJECT_ROOT/"datasets/detection"
OUT_REC            = PROJECT_ROOT/"datasets/recognition"
DATA_DIR           = PROJECT_ROOT/"data"
VAL_RATIO          = 0.2
CLASS_ID           = 0
CLASS_NAMES        = ["license_plate"]

# ───── 1) 이미지 인덱스 구축 ─────────────────────────────────────
print("⏳ Indexing images under", RAW_FULL_IMG_ROOT)
IMG_IDX = {p.name: p for p in RAW_FULL_IMG_ROOT.rglob("*") if p.is_file()}
print(f"📂 Found {len(IMG_IDX)} images")

def find_img(filename: str) -> Path:
    """
    1) 정확히 매칭
    2) 접미사 매칭 (endswith)
    3) 부분 문자열 매칭
    """
    # 1) exact
    if filename in IMG_IDX:
        return IMG_IDX[filename]
    # 2) suffix
    for name, path in IMG_IDX.items():
        if name.endswith(filename):
            return path
    # 3) substring
    for name, path in IMG_IDX.items():
        if filename in name:
            return path
    raise FileNotFoundError(f"{filename} not found")

# ───── 2) 전체 어노테이션 로드 & 셔플 ────────────────────────────
all_ann = []
print("⏳ Loading annotation entries…")
for jf in tqdm(list(RAW_FULL_ANN_DIR.rglob("*.json")), desc="JSON files"):
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except:
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
(OUT_DET / "labels" / "train").mkdir(parents=True, exist_ok=True)
(OUT_DET / "labels" / "val"  ).mkdir(parents=True, exist_ok=True)
OUT_DET.mkdir(parents=True, exist_ok=True)

def yolo_line(bbox, w, h, cid=0):
    (x1,y1),(x2,y2) = bbox
    xc = ((x1+x2)/2)/w; yc = ((y1+y2)/2)/h
    bw = (x2-x1)/w; bh = (y2-y1)/h
    return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def make_detection(entries, subset):
    seen = set()
    list_f = open(OUT_DET/f"{subset}.txt","w")
    for e in tqdm(entries, desc=f"Detect→{subset}", mininterval=1.0):
        fn = Path(e["imagePath"]).name
        if fn in seen:
            continue
        try:
            img_p = find_img(fn)
        except FileNotFoundError:
            continue
        seen.add(fn)

        bbox = e.get("plate",{}).get("bbox")
        if not bbox:
            continue

        w, h = Image.open(img_p).size
        line = yolo_line(bbox, w, h, CLASS_ID)
        # 레이블 파일 쓰기
        lbl_file = OUT_DET/"labels"/subset/f"{img_p.stem}.txt"
        lbl_file.write_text(line + "\n", encoding="utf-8")
        # 리스트에 이미지 경로 추가
        list_f.write(str(img_p.resolve()) + "\n")

    list_f.close()
    print(f"✅ {subset} detection done: {len(seen):,} images")

make_detection(train_entries, "train")
make_detection(val_entries,   "val")

# custom_det.yaml 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
yaml = DATA_DIR/"custom_det.yaml"
yaml.write_text(
    f"train: { (OUT_DET/'train.txt').resolve() }\n"
    f"val:   { (OUT_DET/'val.txt').resolve() }\n\n"
    f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
)
print("✅ Detection config written to", yaml)

# ───── 4) Recognition 전처리 ───────────────────────────────────
plate_jsons = list(RAW_CROP_DIR.rglob("plate.json"))
if not plate_jsons:
    print("⚠️ No plate.json found under", RAW_CROP_DIR, "→ Skipping recognition")
else:
    (OUT_REC / "labels").mkdir(parents=True, exist_ok=True)
    for pj in plate_jsons:
        subset = pj.parent.name  # 예: 'train' 또는 'val'
        img_root = pj.parent/"images"
        out_csv = OUT_REC/"labels"/f"{subset}.csv"
        rows = [["image","plate"]]

        annos = json.loads(pj.read_text(encoding="utf-8"))
        for p in tqdm(annos, desc=f"Recog→{subset}", mininterval=1.0):
            fn  = p.get("imagePath")
            txt = p.get("value","")
            ip  = img_root/fn
            if ip.exists():
                rows.append([str(ip.resolve()), txt])

        with open(out_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerows(rows)
        print(f"✅ {subset} recognition CSV: {len(rows)-1:,} entries")

print("🎉 Preprocessing complete.")
