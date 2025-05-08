import easyocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# EasyOCR 리더 객체 초기화 (한국어 지원)
reader = easyocr.Reader(['ko'])

# 이미지 불러오기
image_path = '32446.jpeg'
image = cv2.imread(image_path)

# OCR 수행
results = reader.readtext(image)

# OpenCV 이미지를 PIL 이미지로 변환 (한글 텍스트 위해)
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

# 한글 폰트 설정 (운영체제에 맞게 경로 수정)
# font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
font_path = "/System/Library/Fonts/AppleGothic.ttf"  # macOS
# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux 예시

font = ImageFont.truetype(font_path, 24)

# 결과 시각화
for (bbox, text, _) in results:
    # 박스 좌표 얻기
    pts = [tuple(map(int, point)) for point in bbox]
    pts = pts + [pts[0]]  # 시작점 다시 추가해서 사각형 완성
    draw.line(pts, fill=(0, 255, 0), width=2)  # 녹색 테두리
    draw.text(pts[0], text, font=font, fill=(0, 0, 255))  # 파란 한글 텍스트

# PIL → OpenCV 이미지로 다시 변환
result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# OCR 텍스트만 추출
lines = [result[1] for result in results]
full_text = ' '.join(lines)

# 텍스트 출력
print("Detected License Plate Text:", full_text)

# 결과 이미지 보여주기
cv2.imshow("Detected Text with Boxes", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
