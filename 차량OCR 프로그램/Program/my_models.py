# models.py
import torch
import numpy as np
import cv2
from PIL import Image, ExifTags, UnidentifiedImageError
from paddleocr import PaddleOCR

# ==============================
# MPS 비활성화 (Mac용 GPU fallback 방지)
# ==============================
def disable_mps():
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

disable_mps()

# ==============================
# 모델 로드
# ==============================
def load_models():
    device = torch.device("cpu")
    # 차량 감지 모델
    car_m = torch.hub.load('./yolov5', 'yolov5s', source='local').to(device)
    # 번호판 감지 모델 (커스텀)
    lp_m = torch.hub.load('./yolov5', 'custom', path='lp_det.pt', source='local').to(device)
    # OCR 모델
    ocr_model = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False)
    # 차량 클래스만 필터링 (2: car, 3: motorcycle 등)
    car_m.classes = [2, 3, 5, 7]
    return car_m, lp_m, ocr_model

# ==============================
# OCR/번호판 추론 함수
# ==============================
def deskew_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is None:
        return plate_img
    angles = [(theta*180/np.pi - 90) for rho, theta in (l[0] for l in lines) if -45 < theta*180/np.pi-90 < 45]
    if not angles:
        return plate_img
    median_angle = np.median(angles)
    h, w = plate_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def group_by_chars(ocr_result, y_thresh=10):
    lines = []
    for box, (text, _) in ocr_result:
        y_center = (box[0][1] + box[2][1]) / 2
        matched = False
        for line in lines:
            if abs(line[0][0] - y_center) < y_thresh:
                line.extend([(y_center, box, ch) for ch in text])
                matched = True
                break
        if not matched:
            lines.append([(y_center, box, ch) for ch in text])
    lines.sort(key=lambda x: x[0][0])
    sorted_texts = []
    for line in lines:
        line.sort(key=lambda x: x[1][0][0])
        sorted_texts.append(''.join([t[2] for t in line]))
    return sorted_texts

def detect_car_plate(img_path, car_m, lp_m, ocr_model):
    """
    번호판 인식 + 번호판 크롭 이미지 리턴
    """
    try:
        im_pil = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return ["인식 불가 (이미지 파일 아님)"], []

    img = np.array(im_pil)
    result_text = []
    plate_imgs = []

    locs = car_m(im_pil).xyxy[0]
    if len(locs) > 0:
        for item in locs:
            x1, y1, x2, y2 = [int(t.cpu().detach().numpy()) for t in item[:4]]
            car_crop = img[y1:y2, x1:x2, :].copy()
            lp_results = lp_m(Image.fromarray(car_crop))
            for lp in lp_results.xyxy[0]:
                lx1, ly1, lx2, ly2 = [int(t.cpu().detach().numpy()) for t in lp[:4]]
                plate_crop = car_crop[ly1:ly2, lx1:lx2].copy()
                # deskew 후 gray 변환
                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                plate_imgs.append(gray_plate)
                ocr_results = ocr_model.ocr(cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR), cls=True)
                if ocr_results and ocr_results[0]:
                    text = ''.join([t[1][0] for t in ocr_results[0]]).replace(',', '').replace('-', '')
                    result_text.append(text)

    if not result_text:
        # 전체 이미지 OCR fallback
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr_full = ocr_model.ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cls=True)
        if ocr_full and ocr_full[0]:
            result_text = [''.join([t[1][0] for t in ocr_full[0]]).replace(',', '').replace('-', '')]
        else:
            result_text = ["인식 실패"]

    return result_text, plate_imgs
