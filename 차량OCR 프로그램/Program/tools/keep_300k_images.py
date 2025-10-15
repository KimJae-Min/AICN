# keep_300k_images.py  (즉시 삭제 모드 지원)
import os, sys, random, argparse, glob
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp"}

def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return [Path(p) for p in files]

def find_label(json_roots, stem):
    # 동일 스템의 .json 라벨을 찾음 (있을 때만)
    for jr in json_roots:
        cand = list(Path(jr).rglob(f"{stem}.json"))
        if cand:
            return cand[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)     # 원본이미지 폴더
    ap.add_argument("--num_keep", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--apply", action="store_true")     # 실제 실행
    ap.add_argument("--labels_root", nargs="*", default=[], help="라벨 JSON 폴더(여러 개 가능)")
    ap.add_argument("--hard_delete", action="store_true", help="바로 영구 삭제")
    args = ap.parse_args()

    imgs = list_images(args.images_root)
    total = len(imgs)
    print(f"[INFO] found images: {total}")
    if total == 0:
        print("이미지 없음."); sys.exit(1)
    if args.num_keep >= total:
        print("[INFO] num_keep >= total. 줄일 필요 없음."); return

    random.seed(args.seed)
    random.shuffle(imgs)
    keep = set(imgs[:args.num_keep])
    drop = imgs[args.num_keep:]
    print(f"[PLAN] keep: {len(keep)} | DROP(삭제): {len(drop)}")

    # 계획 파일 저장(용량 거의 안 듦)
    Path("keep_list.txt").write_text("\n".join(map(str, sor
