import os, random
import cv2
import numpy as np

def image_augmentation(img, ang_range=6, shear_range=3, trans_range=3):
    rows, cols, ch = img.shape

    # Padding으로 외곽 여백 확보
    padded_img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    Rot_M = cv2.getRotationMatrix2D((cols / 2 + 50, rows / 2 + 50), ang_rot, 0.9)
    padded_img = cv2.warpAffine(padded_img, Rot_M, (cols + 100, rows + 100))

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    padded_img = cv2.warpAffine(padded_img, Trans_M, (cols + 100, rows + 100))

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    padded_img = cv2.warpAffine(padded_img, shear_M, (cols + 100, rows + 100))

    # 중앙 Crop
    start_y = (padded_img.shape[0] - rows) // 2
    start_x = (padded_img.shape[1] - cols) // 2
    img = padded_img[start_y:start_y + rows, start_x:start_x + cols]

    # 밝기 조절
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = 0.4 + np.random.uniform()
    img[:, :, 2] *= random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Blur
    blur_value = random.randint(0, 5) * 2 + 1
    img = cv2.blur(img, (blur_value, blur_value))

    return img

# ========== 증강 적용 ========== #
input_dir = "./images/plates/white/"
output_dir = "./images/plates/white_aug/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"{img_path} 를 불러올 수 없습니다.")
            continue

        aug_img = image_augmentation(img)
        save_path = os.path.join(output_dir, "aug_" + filename)
        cv2.imwrite(save_path, aug_img)
        print(f"저장됨: {save_path}")
