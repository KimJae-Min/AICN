import os
import json
import random
import cv2
import numpy as np
from PIL import Image

# ==== 사용자 설정 ====
CAR_IMAGE_ROOT = "images/cars/SUV/BMW"
PLATE_IMAGE_DIR = "images/plates"  # 기존 번호판 이미지 30개가 있는 경로
OUTPUT_IMAGE_DIR = "images/synthetic"
OUTPUT_LABEL_DIR = "labels/synthetic"
OUTPUT_JSON_DIR = "labels/synthetic_json"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# ==== 바운딩 박스 YOLO 포맷 변환 ====
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

# ==== 번호판 위치 감지 ====
def detect_plate_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidate = None
    max_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)

        if 2 < aspect_ratio < 6 and 1000 < area < 30000 and y > img.shape[0] // 2:
            if area > max_area:
                max_area = area
                candidate = (x, y, x + w, y + h)

    return candidate

# ==== 메인 합성 함수 ====
def generate_synthetic_json():
    plate_files = [os.path.join(PLATE_IMAGE_DIR, f) for f in os.listdir(PLATE_IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    if not plate_files:
        print("❌ 번호판 이미지가 없습니다. plates 디렉토리를 확인해주세요.")
        return

    image_files = [f for f in os.listdir(CAR_IMAGE_ROOT) if f.lower().endswith(('.jpg', '.png'))]
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(CAR_IMAGE_ROOT, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] 이미지 로딩 실패: {image_path}")
            continue

        h, w = img.shape[:2]
        plate_bbox = detect_plate_bbox(img)
        if plate_bbox is None:
            print(f"[!] 번호판 감지 실패: {image_file}")
            continue

        x1, y1, x2, y2 = plate_bbox
        pw, ph = x2 - x1, y2 - y1

        plate_path = random.choice(plate_files)
        plate_img = np.array(Image.open(plate_path).convert("RGB"))[:, :, ::-1]
        if plate_img is None:
            print(f"[!] 번호판 이미지 로딩 실패: {plate_path}")
            continue

        try:
            plate_resized = cv2.resize(plate_img, (pw, ph), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[!] 번호판 크기 조정 실패: {image_file} → {e}")
            continue

        img[y1:y2, x1:x2] = plate_resized

        output_img_path = os.path.join(OUTPUT_IMAGE_DIR, image_file)
        cv2.imwrite(output_img_path, img)

        yolo_box = convert_to_yolo((w, h), (x1, y1, x2, y2))
        label_path = os.path.join(OUTPUT_LABEL_DIR, image_file.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

        json_label = {
            "imagePath": os.path.join("SUV/BMW", image_file),
            "car": {
                "bbox": [[0, 0], [w, h]]
            },
            "plate": {
                "image": os.path.basename(plate_path),
                "bbox": [[x1, y1], [x2, y2]]
            }
        }
        json_path = os.path.join(OUTPUT_JSON_DIR, image_file.replace('.jpg', '.json'))
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(json_label, jf, ensure_ascii=False, indent=2)

        print(f"[{idx + 1}/{len(image_files)}] 완료: {image_file} → 번호판 이미지: {os.path.basename(plate_path)}")

    print("✅ 모든 번호판 합성과 JSON 생성 완료!")

if __name__ == "__main__":
    generate_synthetic_json()
