import os
import zipfile
import uuid
from pathlib import Path
from datetime import datetime
import streamlit as st
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError

from my_models import load_models, detect_car_plate

# ==============================
# Streamlit 페이지 설정
# ==============================
st.set_page_config(layout='wide')

# ==============================
# 파일 처리 함수
# ==============================
def save_uploaded_file(directory, uploaded_file):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, uploaded_file.name)
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return path


def extract_zip(zip_path):
    extract_to = f"extracted/{uuid.uuid4().hex}"
    os.makedirs(extract_to, exist_ok=True)
    files = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    for root, _, filenames in os.walk(extract_to):
        for fn in filenames:
            full_path = os.path.join(root, fn)
            if "__MACOSX" in full_path or fn.startswith("._"):
                continue
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(full_path)
            elif full_path.lower().endswith('.zip'):
                files += extract_zip(full_path)
    return files


def get_image_date(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                name = ExifTags.TAGS.get(tag)
                if name in ('DateTimeOriginal', 'DateTime'):
                    return datetime.strptime(val, '%Y:%m:%d %H:%M:%S')
    except:
        pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def save_to_excel(infos, filename):
    if not infos:
        st.warning("⚠️ 내보낼 데이터가 없습니다.")
        return

    os.makedirs('excel_outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"excel_outputs/{filename}_{timestamp}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "차량 인식 결과"
    ws.append(["촬영일", "파일명", "차량 번호", "이미지 미리보기"])
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 20
    ws.column_dimensions['D'].width = 40

    for idx, info in enumerate(infos, start=2):
        ws.cell(idx, 1, info['capture_time'])
        ws.cell(idx, 2, info['name'])
        ws.cell(idx, 3, info['plate'])
        try:
            for p_img in info['plate_imgs']:
                img = ExcelImage(p_img)
                img.width = 150
                img.height = int(150 * img.height / img.width)
                ws.add_image(img, f"D{idx}")
                ws.row_dimensions[idx].height = img.height * 0.75
        except:
            continue

    wb.save(out_path)
    st.success(f"📊 엑셀 저장 완료: `{out_path}`")


# ==============================
# Streamlit 모델 로드
# ==============================
@st.cache_resource
def load_models_cached():
    return load_models()


# ==============================
# Main
# ==============================
def main():
    plate_detector, ocr_bundle = load_models_cached()

    if 'file_info' not in st.session_state:
        st.session_state['file_info'] = []
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = set()
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1

    st.title("🚗 차량 번호판 자동 인식 시스템")
    menu = ['📂 파일 업로드', '🔧 번호판 수정 및 결과 확인', 'ℹ️ About']
    choice = st.sidebar.radio('메뉴 선택', menu)

    # ==============================
    # 📂 파일 업로드
    # ==============================
    if choice == '📂 파일 업로드':
        col_upload, col_list = st.columns([1, 2])

        # 좌측 업로드
        with col_upload:
            st.markdown("### 📁 이미지 / ZIP 파일 업로드")
            st.markdown("- 이미지(.png, .jpg, .jpeg) 또는 ZIP(.zip)")

            uploaded = st.file_uploader(
                "파일 선택 👇",
                type=['png', 'jpg', 'jpeg', 'zip'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            log_box = st.empty()
            progress_bar = st.progress(0)
            infos = []

            if uploaded:
                for idx, f in enumerate(uploaded, start=1):
                    progress_bar.progress(idx / len(uploaded))
                    log_box.info(f"📄 {f.name} 처리 중...")

                    if f.name in st.session_state['processed_files']:
                        log_box.warning(f"⚠️ 이미 처리된 파일: {f.name} → 건너뜀")
                        continue

                    saved_path = save_uploaded_file('uploads', f)
                    files_to_process = [saved_path]
                    if f.name.lower().endswith('.zip'):
                        files_to_process = extract_zip(saved_path)

                    for fp in files_to_process:
                        file_key = Path(fp).name
                        if file_key in st.session_state['processed_files']:
                            log_box.warning(f"⚠️ 이미 처리된 파일: {file_key} → 건너뜀")
                            continue
                        st.session_state['processed_files'].add(file_key)

                        plates, plate_imgs = detect_car_plate(fp, plate_detector, ocr_bundle)
                        infos.append({
                            'capture_time': get_image_date(fp).strftime('%Y-%m-%d %H:%M:%S'),
                            'name': file_key,
                            'plate': ", ".join(plates),
                            'path': fp,
                            'plate_imgs': plate_imgs
                        })

                if infos:
                    st.session_state['file_info'] += infos
                    st.session_state['file_info'].sort(
                        key=lambda x: datetime.strptime(x['capture_time'], '%Y-%m-%d %H:%M:%S')
                    )

                progress_bar.progress(1.0)
                log_box.success("✅ 업로드 및 인식 완료!")

        # 우측 차량 목록
        with col_list:
            if st.session_state['file_info']:
                st.markdown("### 📸 인식된 차량 목록")
                for info in st.session_state['file_info']:
                    st.write(f"📅 {info['capture_time']} | 🚘 {info['plate']} | 🖼 {info['name']}")

    # ==============================
    # 🔧 번호판 수정 및 결과 확인
    # ==============================
    elif choice == '🔧 번호판 수정 및 결과 확인':
        if not st.session_state['file_info']:
            st.info("📁 먼저 파일을 업로드하세요.")
            return

        file_info = st.session_state['file_info']
        per_page = 10
        total_pages = (len(file_info) + per_page - 1) // per_page

        current_page = st.session_state['current_page']
        start_idx = (current_page - 1) * per_page
        end_idx = start_idx + per_page
        current_items = file_info[start_idx:end_idx]

        # 좌우 분할
        left_col, right_col = st.columns([1, 2])

        # 좌측: 차량 선택 라디오
        with left_col:
            st.markdown("### 차량 목록")
            selected_plate_key = st.radio(
                "선택할 차량을 클릭하세요",
                options=[f"{i+start_idx+1}. {info['plate']} ({info['name']})" for i, info in enumerate(current_items)],
                key='selected_plate_radio'
            )
            selected_idx = int(selected_plate_key.split('.')[0]) - 1
            info = file_info[selected_idx]

        # 우측: 원본 이미지 + 첫 번째 번호판 크롭
        with right_col:
            st.markdown("### 선택한 차량 상세보기")
            orig_img = info['path']
            plate_imgs = info['plate_imgs']

            img_cols = st.columns(2)
            # 원본 이미지
            with img_cols[0]:
                st.image(orig_img, caption="📷 원본 이미지", use_container_width=True)
            # 첫 번째 번호판 크롭
            if plate_imgs:
                with img_cols[1]:
                    st.image(plate_imgs[0], caption="🔍 번호판 크롭 1", use_container_width=True)

            # 번호판 수정
            def update_plate(path):
                for item in st.session_state['file_info']:
                    if item['path'] == path:
                        item['plate'] = st.session_state[f"edit_{path}"]

            st.text_input(
                "번호판 수정",
                value=info['plate'],
                key=f"edit_{info['path']}",
                on_change=update_plate,
                args=(info['path'],)
            )

        # 페이지네이션
        st.markdown("---")
        st.markdown("#### 📄 페이지 선택")
        col_prev, col_page, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("◀ 이전") and current_page > 1:
                st.session_state['current_page'] -= 1
                st.rerun()

        with col_page:
            st.markdown(f"### {current_page} / {total_pages}", unsafe_allow_html=True)

        with col_next:
            if st.button("다음 ▶") and current_page < total_pages:
                st.session_state['current_page'] += 1
                st.rerun()

        # 엑셀 내보내기
        st.markdown("---")
        fn = st.text_input("엑셀 파일명", "vehicles")
        if st.button("📊 엑셀로 내보내기"):
            save_to_excel(file_info, fn)

    # ==============================
    # ℹ️ About
    # ==============================
    elif choice == 'ℹ️ About':
        st.markdown("### 시스템 개요")
        st.write("""
        YOLOv5 + PaddleOCR 기반 차량 번호판 인식 시스템입니다.  
        이미지를 업로드하면 자동으로 번호판을 인식하고, 결과를 Excel로 내보낼 수 있습니다.
        """)


if __name__ == '__main__':
    main()
