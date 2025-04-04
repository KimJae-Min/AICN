import streamlit as st
import zipfile
import os
import torch
import easyocr
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
from datetime import datetime

st.set_page_config(layout='wide')  # 페이지 레이아웃 설정

# 파일 저장 함수
def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

# ZIP 파일 압축 해제 함수
def extract_zip(zip_path, extract_to='extracted'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)  # ZIP 내부 파일 압축 해제
    return extract_to  # 압축 해제된 폴더 경로 반환

# 엑셀 저장 함수
def save_to_excel(file_info, file_name):
    output_dir = "excel_outputs"  # 저장할 디렉토리 지정
    if not os.path.exists(output_dir):  # 폴더가 없으면 생성
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{file_name}.xlsx")  # 입력한 파일명으로 저장

    df = pd.DataFrame(file_info, columns=['파일명', '차량 번호', '파일 시간'])  # 순서 지정
    df.to_excel(output_path, index=False)
    st.success(f"엑셀 파일이 '{output_path}'로 저장되었습니다.")

def main():
    car_m, lp_m, reader = load_model()  # 모델 로드
    st.title('이미지 또는 ZIP 파일을 등록하세요')

    menu = ['About', '파일 업로드', '공란']
    choice = st.sidebar.selectbox('메뉴', menu)

    if choice == 'About':
        st.subheader('안내 페이지')

    elif choice == '파일 업로드':
        st.subheader('이미지 또는 ZIP 파일 업로드')

        uploaded_files = st.file_uploader('이미지를 업로드하거나 ZIP 파일을 선택하세요.', 
                                          type=['png', 'jpg', 'jpeg', 'zip'], 
                                          accept_multiple_files=True)

        file_info = []  # 업로드된 파일 정보를 저장할 리스트

        if uploaded_files:
            for uploaded_file in uploaded_files:
                current_time = datetime.now().isoformat().replace(':', "_")  # 현재 시간
                file_ext = uploaded_file.name.split('.')[-1].lower()

                if file_ext == 'zip':  # ZIP 파일 처리
                    zip_path = save_uploaded_file('uploads', uploaded_file)
                    extracted_folder = extract_zip(zip_path)  # 압축 해제
                    
                    for root, _, files in os.walk(extracted_folder):
                        for file_name in files:
                            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                                file_path = os.path.join(root, file_name)
                                im, text = detect(car_m, lp_m, reader, file_path)  # 이미지 처리
                                st.write(text)  # 인식된 번호판 출력
                                st.image(im)  # 결과 이미지 출력

                                license_plate = ", ".join(text) if text else "인식 실패"

                                file_info.append({
                                    '파일명': file_name,
                                    '차량 번호': license_plate,
                                    '파일 시간': current_time
                                })
                
                else:  # 개별 이미지 처리
                    file_path = save_uploaded_file('uploads', uploaded_file)
                    im, text = detect(car_m, lp_m, reader, file_path)  # 이미지 처리
                    st.write(text)  # 인식된 번호판 출력
                    st.image(im)  # 결과 이미지 출력

                    license_plate = ", ".join(text) if text else "인식 실패"

                    file_info.append({
                        '파일명': uploaded_file.name,
                        '차량 번호': license_plate,
                        '파일 시간': current_time
                    })

            # 파일명 입력, 파일 저장장
            file_name = st.text_input("저장할 엑셀 파일명을 입력하세요 (확장자 제외)", "uploaded_images_info")

            if st.button("Save"):
                save_to_excel(file_info, file_name)

    else:
        st.subheader('공란')

@st.cache  # Streamlit 캐싱 기능 (모델 로드를 매번 하지 않도록)

def load_model():
    car_m = torch.hub.load("ultralytics/yolov5", 'yolov5s', force_reload=True, skip_validation=True)  # YOLOv5 차량 탐지 모델
    lp_m = torch.hub.load('ultralytics/yolov5', 'custom', 'lp_det.pt')  # 사용자 정의 번호판 탐지 모델
    reader = easyocr.Reader(['en'], detect_network='craft', recog_network='best_acc', 
                            user_network_directory='lp_models/user_network', 
                            model_storage_directory='lp_models/models')  # EasyOCR 설정
    
    car_m.classes = [2, 3, 5, 7]  # 차량 클래스 (자동차, 트럭 등) 설정
    return car_m, lp_m, reader  # 모델 반환

def detect(car_m, lp_m, reader, path):
    fontpath = "SpoqaHanSansNeo-Light.ttf"  # 폰트 파일 경로
    font = ImageFont.truetype(fontpath, 200)  # 폰트 설정
    im = Image.open(path)  # 업로드된 이미지 열기
    to_draw = np.array(im)  # 이미지를 NumPy 배열로 변환
    results = car_m(im)  # 차량 탐지 실행
    locs = results.xyxy[0]  # 차량 좌표 가져오기
    result_text = []  # 인식된 번호판 텍스트 저장 리스트

    if len(locs) == 0:  # 차량이 검출되지 않은 경우
        return cv2.resize(to_draw, (1280, 1280)), ["검출된 차 없음"]

    for idx, item in enumerate(locs):  # 차량이 검출된 경우 실행
        x, y, x1, y1 = [int(it.cpu().detach().numpy()) for it in item[:4]]
        car_im = to_draw[y:y1, x:x1, :].copy()
        result = lp_m(Image.fromarray(car_im))  # 번호판 탐지 실행

        for rslt in result.xyxy[0]:  
            x2, y2, x3, y3 = [int(it.cpu().detach().numpy()) for it in rslt[:4]]
            im_crop = cv2.cvtColor(cv2.resize(to_draw[y+y2:y+y3, x+x2:x+x3], (224, 128)), cv2.COLOR_BGR2GRAY)
            text = reader.recognize(im_crop)[0][1] if reader.recognize(im_crop) else "인식 실패"
            result_text.append(text)

            img_pil = Image.fromarray(to_draw)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x+x2-100, y+y2-300), text, font=font, fill=(255, 0, 255))
            to_draw = np.array(img_pil)
            cv2.rectangle(to_draw, (x+x2, y+y2), (x+x3, y+y3), (255, 0, 255), 10)

    return cv2.resize(to_draw, (1280, 1280)), result_text

if __name__ == '__main__':
    main()
