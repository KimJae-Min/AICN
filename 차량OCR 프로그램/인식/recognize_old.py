import easyocr
import cv2

# EasyOCR 리더 객체 초기화 (한국어 지원)
reader = easyocr.Reader(['ko'])

image_path = '56789.png'
image = cv2.imread(image_path)

# OCR 수행 (이미지에서 텍스트 인식)
results = reader.readtext(image)

# 결과에서 텍스트 부분만 추출
lines = [result[1] for result in results]  # OCR 결과에서 텍스트만 추출

# 번호판의 상단과 하단 텍스트 구분 후 합치기
full_text = ' '.join(lines)  # 두 줄을 공백으로 합침

# 결과 출력
print("Detected License Plate Text:", full_text)
