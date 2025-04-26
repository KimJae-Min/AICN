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

## 실행 방법
/*create_plate.py*/
  - 생성하는 번호판 종류
    - 신형 8자리 번호판 (홀로그램)
    - 구형 8자리 번호판
    - 구형 7자리 번호판
  - 한글 문자 셋 설정
    - 차량별 번호판에 사용되는 한글 문자 분류
      - korean: 일반 차량 번호판에 사용할 한글 문자들
      - korean_taxi: 택시용 차량 번호판에 사용할 한글 문자들
