import os
import json
import random
import cv2
import numpy as np
from PIL import Image

# ==== 사용자 설정 ====
JSON_LABEL_DIR = "labels/filtered_label"        # JSON 라벨 파일들이 있는 디렉토리
CAR_IMAGE_ROOT = "images/cars/SUV/BMW"          # 차량 이미지들이 저장되어 있는 디렉토리
PLATE_IMAGE_DIR = "images/plates"               # 사용할 번호판 이미지들이 저장되어 있는 디렉토리
OUTPUT_IMAGE_DIR = "images/synthetic"           # 최종 합성된 차량 이미지가 저장되는 디렉토리
OUTPUT_LABEL_DIR = "labels/synthetic"           # 합성된 차량 이미지에 대한 YOLO 포맷 라벨을 저장할 디렉토리

# ==== 디렉토리 준비 ====
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ==== 바운딩 박스 YOLO 포맷 변환 ====
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0                 # 중심 x좌표
    y = (box[1] + box[3]) / 2.0                 # 중심 y좌표
    w = box[2] - box[0]                         # box' width
    h = box[3] - box[1]                         # box' height
    return (x * dw, y * dh, w * dw, h * dh)

# ==== 메인 합성 함수 ====
def generate_synthetic_from_json():
    plate_files = [os.path.join(PLATE_IMAGE_DIR, f) for f in os.listdir(PLATE_IMAGE_DIR) if f.endswith(('.png', '.jpg'))]       # 번호판 이미지 파일 목록 가져오기
    
    if not plate_files:
        print("❌ 번호판 이미지가 없습니다. images/plates 디렉토리를 확인하세요.")
        return

    json_files = [f for f in os.listdir(JSON_LABEL_DIR) if f.endswith(".json")]     # JSON 파일 목록 가져오기
    
    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(JSON_LABEL_DIR, json_file)     # JSON 파일 열기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = data['imagePath']
        image_path = os.path.join(CAR_IMAGE_ROOT, os.path.basename(image_name))

        # 이미지가 존재하지 않는 경우 -> 건너 뛰기
        if not os.path.exists(image_path):
            print(f"[!] 이미지 없음: {image_path}")
            continue

        # 이미지 읽기
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] 이미지 로딩 실패: {image_path}")
            continue
        
        # 현재 차량 이미지의 크기
        h, w = img.shape[:2]

        try:
            # 차량 bbox (원본 이미지 기준)
            car_pts = data['car']['bbox']
            cx1 = int(round(float(car_pts[0][0])))
            cy1 = int(round(float(car_pts[0][1])))
            cx2 = int(round(float(car_pts[1][0])))
            cy2 = int(round(float(car_pts[1][1])))
            
            # 좌표 정렬
            if cx1 > cx2: cx1, cx2 = cx2, cx1
            if cy1 > cy2: cy1, cy2 = cy2, cy1

            car_w = cx2 - cx1
            car_h = cy2 - cy1

            # 번호판 bbox (원본 이미지 기준)
            pts = data['plate']['bbox']
            plate_x1 = int(round(float(pts[0][0])))
            plate_y1 = int(round(float(pts[0][1])))
            plate_x2 = int(round(float(pts[1][0])))
            plate_y2 = int(round(float(pts[1][1])))
            
            # 좌표 정렬
            if plate_x1 > plate_x2: plate_x1, plate_x2 = plate_x2, plate_x1
            if plate_y1 > plate_y2: plate_y1, plate_y2 = plate_y2, plate_y1

        except:
            print(f"[!] 번호판 정보 없음 또는 형식 오류: {image_name}")
            continue

        # 번호판 bbox를 차량 bbox 기준 상대 좌표로 변환 
        new_x1 = plate_x1 - cx1
        new_y1 = plate_y1 - cy1
        new_x2 = plate_x2 - cx1
        new_y2 = plate_y2 - cy1

        print(f"[디버깅] {image_name} → 차량크기({w}x{h}) / plate 상대좌표: ({new_x1},{new_y1})~({new_x2},{new_y2})")

        # 번호판 bbox의 너비, 높이 계산
        pw = new_x2 - new_x1
        ph = new_y2 - new_y1

        # 번호판이 너무 작은 경우 -> 스킵
        if pw <= 10 or ph <= 5:
            print(f"[!] 번호판 bbox 너무 작음: {image_name}")
            continue

        try:
            # 번호판 이미지 랜덤 선택
            plate_path = random.choice(plate_files)
            plate_img = Image.open(plate_path).convert("RGB")
            plate_img = np.array(plate_img)

            # 번호판 크기 리사이즈
            plate_resized = cv2.resize(plate_img, (pw, ph), interpolation=cv2.INTER_AREA)

            # 차량 이미지에 번호판 삽입
            img[new_y1:new_y2, new_x1:new_x2] = plate_resized

        except Exception as e:
            print(f"[!] 번호판 처리 실패: {image_name} → {e}")
            continue
        
        # 결과 이미지 저장 (OUTPUT_IMAGE_DIR)
        output_img_path = os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(image_name))
        cv2.imwrite(output_img_path, img)

        # YOLO 라벨 파일 저장
        yolo_box = convert_to_yolo((w, h), (new_x1, new_y1, new_x2, new_y2))
        label_path = os.path.join(OUTPUT_LABEL_DIR, os.path.basename(image_name).replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

        if idx % 50 == 0:
            print(f"[{idx}/{len(json_files)}] 완료: {image_name} → 부착된 번호판: {os.path.basename(plate_path)}")

    print("✅ 모든 차량 이미지에 번호판 합성 완료!")

# ==== 실행 ====
if __name__ == "__main__":
    generate_synthetic_from_json()
