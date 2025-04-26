# 📖 README

## 📌 목적
- 학습용 **가상 한국 번호판 이미지 생성기**.

---

## 🛠️ 필요한 환경
- Python
- 필요 라이브러리 설치
  ```bash
  pip install tqdm pillow colorama
  ```
- 라이브러리 설명
  - **tqdm**: 진행바 표시
  - **Pillow (PIL)**: 이미지 생성 및 편집
  - **urllib.request**: 배경 이미지 다운로드

---

## 📂 파일 설명
- `generate_img.py`  
  → 생성한 번호판 이미지를 차량 이미지에 붙여내는 코드
- `create_plate.py`  
  → 랜덤 번호판 이미지를 생성하는 코드
- 사용된 구글
  - `Hangil.ttf`: 한글용 포트
  - `NotoSansKR-Medium.ttf`: 숫자용 포트

---

## 🚀 실행 방법

### ▶️ create_plate.py
**흐름 요약**

1. 번호판 종류 결정
2. 랜덤 글자 조합 (숫자 + 한글 + 숫자)
3. 배경 이미지 열기
4. 글자 위치에 텍스트 그린 후 저장

---

**생성 가능한 번호판 종류**
- 신형 8자리 번호판 (홀로그램 포함)
- 구형 8자리 번호판
- 구형 7자리 번호판

---

**한글 문자셀 구성**
- `korean`: 일반 차량 번호판용 한글
- `korean_taxi`: 택시용 한글
- `korean_rent`: 렌터카용 한글
- `korean_parcel`: 택배 차량용 한글



---

**배경 이미지 파일**
- `images/src_plate_img/number_plate_new.png`: 신형 배경
- `images/src_plate_img/number_plate_old.png`: 구형 배경

---

**번호판 문자열 조합 방법**
- 아파 (숫자) + 중간자리 (한글) + 뒤자리 (숫자)
- 이어서 전체 번호판 문자열 생성 후, 배경에 텍스트 그린 후 저장

---

### ▶️ generate_img.py
**흐름 요약**
(*create_plate.py에서 생성한 이미지를 사용함)
1. `plates/` 에서 번호판 이미지 불러오기
2. `labels/filtered_label/` 에서 JSON 파일 불러오기
3. 각 차량에 대해:
   - JSON에서 차량 bbox, 번호판 bbox 읽기
4. 차량 이미지 열기
5. 번호판 이미지 붙여기 (랜덤 선택 + 리사이즈)
6. 결과 이미지 저장

---

**메인 함수: `generate_synthetic_from_json()`**

- 사용할 번호판 이미지 불러오기
  ```python
  plate_files = [
      os.path.join(PLATE_IMAGE_DIR, f)
      for f in os.listdir(PLATE_IMAGE_DIR)
      if f.endswith(('.png', '.jpg'))
  ]
  ```

- 사용할 JSON 파일 목록 불러오기
  ```python
  json_files = [
      f for f in os.listdir(JSON_LABEL_DIR)
      if f.endswith(".json")
  ]
  ```

- 차량 bbox + 번호판 bbox 출력
  ```python
  car_pts = data['car']['bbox']
  plate_pts = data['plate']['bbox']
  ```
  - (x1, y1), (x2, y2) 형식으로 정리
  - 조합 순서 정렬 (x1<x2, y1<y2)

- 번호판 bbox를 차량 기준 상대좌표로 변환
  ```python
  new_x1 = plate_x1 - cx1
  new_y1 = plate_y1 - cy1
  new_x2 = plate_x2 - cx1
  new_y2 = plate_y2 - cy1
  ```

- 번호판 이미지 붙여기
  ```python
  plate_img = random.choice(plate_files)
  plate_resized = cv2.resize(plate_img, (pw, ph))
  img[new_y1:new_y2, new_x1:new_x2] = plate_resized
  ```

---


