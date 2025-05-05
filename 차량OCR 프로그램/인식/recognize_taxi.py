import cv2
import numpy as np
from paddleocr import PaddleOCR

# PaddleOCR 초기화
ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # use_angle_cls를 True로 설정하면 회전된 텍스트도 인식 가능

def preprocess_and_recognize(image_path):
    # 1. 이미지 불러오기
    image = cv2.imread(image_path)

    # 2. 세로 글씨와 가로 글씨 분할 (왼쪽 1/5은 세로, 오른쪽 4/5은 가로)
    height, width = image.shape[:2]
    vertical_region = image[:, :width // 5]
    horizontal_region = image[:, width // 5:] 

    # 3. 세로 글씨 회전 (세로는 위→아래로 읽히므로, 시계방향 90도 회전)
    vertical_rotated = cv2.rotate(vertical_region, cv2.ROTATE_90_CLOCKWISE)

    # 4. OCR 인식
    result_vertical = ocr.ocr(vertical_rotated, cls=True)
    result_horizontal = ocr.ocr(horizontal_region, cls=True)

    # 5. 결과 텍스트 정리
    vertical_text = ''.join([line[1][0] for line in result_vertical[0]])
    horizontal_text = ''.join([line[1][0] for line in result_horizontal[0]])

    # 6. 전체 차량번호 조합
    full_plate_number = vertical_text + horizontal_text
    return full_plate_number

# 예시 실행
if __name__ == "__main__":
    plate_number = preprocess_and_recognize("12345.png")
    print("차량번호:", plate_number)
