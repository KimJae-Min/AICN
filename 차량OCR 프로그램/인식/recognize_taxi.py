import cv2
import numpy as np
from paddleocr import PaddleOCR

# PaddleOCR 초기화
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

def draw_ocr_boxes(image, ocr_result, color=(0, 255, 0)):
    for line in ocr_result[0]:
        box = np.array(line[0]).astype(int)
        cv2.polylines(image, [box], isClosed=True, color=color, thickness=2)
    return image

def preprocess_and_recognize(image_path):
    # 1. 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 2. 세로 / 가로 영역 분리
    height, width = image.shape[:2]
    vertical_region = image[:, :width // 5]
    horizontal_region = image[:, width // 5:]

    # 3. 세로 글씨 회전
    vertical_rotated = cv2.rotate(vertical_region, cv2.ROTATE_90_CLOCKWISE)

    # 4. OCR 인식
    result_vertical = ocr.ocr(vertical_rotated, cls=True)
    result_horizontal = ocr.ocr(horizontal_region, cls=True)

    # 5. 인식 결과 시각화 (각각 원본 크기로 되돌리기)
    vertical_visual = cv2.rotate(draw_ocr_boxes(vertical_rotated.copy(), result_vertical, color=(255, 0, 0)), cv2.ROTATE_90_COUNTERCLOCKWISE)
    horizontal_visual = draw_ocr_boxes(horizontal_region.copy(), result_horizontal, color=(0, 255, 0))

    # 전체 이미지 복사본 만들기
    visualized_image = np.hstack((vertical_visual, horizontal_visual))

    # 6. 텍스트 추출
    vertical_text = ''
    if result_vertical and result_vertical[0]:
        vertical_text = ''.join([line[1][0] for line in result_vertical[0]])

    horizontal_text = ''
    if result_horizontal and result_horizontal[0]:
        horizontal_text = ''.join([line[1][0] for line in result_horizontal[0]])

    full_plate_number = vertical_text + horizontal_text
    return full_plate_number, visualized_image

# 실행
if __name__ == "__main__":
    try:
        plate_number, annotated_image = preprocess_and_recognize("32445.jpeg")
        print("차량번호:", plate_number)

        # 결과 이미지 저장 또는 보여주기
        cv2.imshow("인식 결과", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 또는 파일로 저장하려면:
        # cv2.imwrite("result_with_boxes.jpg", annotated_image)

    except Exception as e:
        print("오류 발생:", e)
