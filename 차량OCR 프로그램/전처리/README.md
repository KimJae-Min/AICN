## 목적
  - 학습용 가상 한국 번호판 이미지 생성기

## 필요한 환경
  - python
  - 필요한 라이브러리 설치
    ``` bash
      pip install tqdm pillow colorama
    ```
  - tqdm: 진행바 표시
  - PTL (Pillow): 이미지 생성 및 편집
  - urllib.equest: 배경 이미지 다운로드

## 파일 설명
  - ```generate_img.py```: 생성한 이미지를 차량 이미지에 부착한 결과 사진을 저장
  - ```create_plate.py```: 랜덤 번호판 생성
  - 사용한 fonts
      - ```Hangil.ttf```
      - ```NotoSansKR-Medium.ttf```
