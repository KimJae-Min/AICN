# -*- coding: utf-8 -*-
"""
30만 이미지/YOLO txt가 이미 data/images/train, data/labels/train 에 있는 상태에서
라벨 원본 JSON(라벨링 데이터/**.json)을 이용해 image stem -> videoName 매핑을 만들고,
videoName(그룹) 단위로 train/val을 분리한다.

- 전제:
  images/train 에는 이미지(파일명=...jpg)
  labels/train 에는 동명 txt (한 이미지당 1개)
- JSON에는 최소한 {"imagePath": "...jpg", "videoName": "...mp4"}가 있어야 한다.
- 스플릿 비율 또는 목표 val 샘플 수로 제어 가능.
- 기본은 "이동(move)"로 공간 절약. (복사하려면 --copy)
"""

import os, json, glob, random, argparse, shutil
from pathlib import Path
from collections import defaultdict

def scan_train_set(img_dir, lbl_dir):
    imgs = {Path(p).stem: p for p in glob.glob(os.path.join(img_dir, "*")) if os.path.isfile(p)}
    lbls = {Path(p).stem: p for p in glob.glob(os.path.join(lbl_dir, "*")) if os.path.isfile(p)}
    stems = sorted(set(imgs.keys()) & set(lbls.keys()))
    return imgs, lbls, stems

def build_stem_to_video(labels_json_root, stems_set):
    # 라벨 원본 JSON을 훑어서 stem -> videoName 매핑
    # imagePath의 파일명(stem) 기준으로 연결
    mapping = {}
    jpaths = glob.glob(os.path.join(labels_json_root, "**", "*.json"), recursive=True)
    for jp in jpaths:
        try:
            js = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        ip = js.get("imagePath")
        vn = js.get("videoName")
        if not ip or not vn:
            continue
        stem = Path(ip).stem
        if stem in stems_set:
            mapping[stem] = vn
    return mapping

def group_by_video(stems, stem2vid):
    groups = defaultdict(list)
    unknown = []
    for s in stems:
        vn = stem2vid.get(s)
        if vn is None:
            unknown.append(s)
        else:
            groups[vn].append(s)
    return groups, unknown

def choose_val_groups(groups, target_val_count=None, val_ratio=None, seed=42):
    vids = list(groups.keys())
    random.Random(seed).shuffle(vids)
    total = sum(len(groups[v]) for v in vids)
    if target_val_count is None and val_ratio is None:
        val_ratio = 0.1  # default 10%

    if target_val_count is None:
        target_val_count = int(total * val_ratio)

    chosen, cnt = [], 0
    for v in vids:
        if cnt >= target_val_count: break
        chosen.append(v)
        cnt += len(groups[v])
    return set(chosen), cnt, total

def move_or_copy(stems, imgs_map, lbls_map, out_img, out_lbl, move=True):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_lbl).mkdir(parents=True, exist_ok=True)
    imoved = lmoved = 0
    for s in stems:
        si = imgs_map[s]; sl = lbls_map[s]
        di = os.path.join(out_img, os.path.basename(si))
        dl = os.path.join(out_lbl, os.path.basename(sl))
        if move:
            shutil.move(si, di); shutil.move(sl, dl)
        else:
            shutil.copy2(si, di); shutil.copy2(sl, dl)
        imoved += 1; lmoved += 1
    return imoved, lmoved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img", required=True, help="data/images/train")
    ap.add_argument("--train_lbl", required=True, help="data/labels/train")
    ap.add_argument("--labels_json_root", required=True, help="라벨링 데이터 (원본 json 루트)")
    ap.add_argument("--out_val_img", required=True, help="data/images/val")
    ap.add_argument("--out_val_lbl", required=True, help="data/labels/val")
    ap.add_argument("--val_ratio", type=float, default=None, help="예: 0.1 (10%)")
    ap.add_argument("--val_count", type=int, default=None, help="목표 val 이미지 수(우선)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="기본은 move(이동). 공간 충분하면 복사")
    args = ap.parse_args()

    imgs_map, lbls_map, stems = scan_train_set(args.train_img, args.train_lbl)
    stems_set = set(stems)
    print(f"[INFO] train stems: {len(stems)}")

    stem2vid = build_stem_to_video(args.labels_json_root, stems_set)
    groups, unknown = group_by_video(stems, stem2vid)
    if unknown:
        print(f"[WARN] videoName 못 찾은 stem: {len(unknown)} (임의로 train 유지)")

    chosen_vids, picked, total = choose_val_groups(groups,
        target_val_count=args.val_count, val_ratio=args.val_ratio, seed=args.seed)

    # val로 갈 stem 목록
    val_stems = []
    for v in chosen_vids:
        val_stems.extend(groups[v])
    val_stems = set(val_stems)

    print(f"[PLAN] total={total} | val={len(val_stems)} (목표={args.val_count or int(total*(args.val_ratio or 0.1))}) | "
          f"videos_selected={len(chosen_vids)}")

    # 이동/복사 실행
    imoved, lmoved = move_or_copy(val_stems, imgs_map, lbls_map,
                                  args.out_val_img, args.out_val_lbl,
                                  move=(not args.copy))
    print(f"[DONE] images->{args.out_val_img}: {imoved}, labels->{args.out_val_lbl}: {lmoved}")
    # 남은 train 개수는 폴더에서 확인하면 됨.

if __name__ == "__main__":
    main()
