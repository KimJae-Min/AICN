import os, cv2, math, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def imread_unicode(path, flag=cv2.IMREAD_GRAYSCALE):
    # Windows에서 한글/공백/긴 경로도 안전하게 읽기
    path = os.path.normpath(path)
    data = np.fromfile(path, dtype=np.uint8)   
    img  = cv2.imdecode(data, flag)           
    return img

def resize_keep_ratio_pad(img, H=64, W=256):
    h, w = img.shape[:2]
    scale = H / max(1, h)
    nw = min(W, int(round(w * scale)))
    resized = cv2.resize(img, (nw, H), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((H, W), dtype=resized.dtype)
    canvas[:, :nw] = resized
    return canvas

def auto_gamma(img, low=1.5, high=0.6):
    # 히스토그램으로 밝기 추정해 어두우면 gamma<1, 밝으면 >1
    m = np.mean(img) / 255.0
    gamma = low if m < 0.35 else (high if m > 0.65 else 1.0)
    lut = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)

def clahe(img, clip=2.0, tiles=(8,8)):
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return cl.apply(img)

def denoise(img, method="bilateral"):
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    elif method == "nlm":
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        return img

def unsharp(img, k=1.0, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    sharp = cv2.addWeighted(img, 1+k, blur, -k, 0)
    return sharp

def sauvola_binarize(img, win=25, k=0.2):
    # OpenCV만으로 근사 구현 (간단 버전)
    mean = cv2.boxFilter(img, ddepth=cv2.CV_32F, ksize=(win,win))
    sqmean = cv2.boxFilter((img*img).astype(np.float32), ddepth=cv2.CV_32F, ksize=(win,win))
    var = np.maximum(sqmean - mean*mean, 0.0)
    std = np.sqrt(var)
    R = 128.0
    thresh = mean*(1 + k*((std/R)-1))
    bin_img = (img.astype(np.float32) > thresh).astype(np.uint8)*255
    return bin_img

_SR = None 

def super_res_adaptive(img_gray, target_h, max_scale=2.5):
    h = max(1, img_gray.shape[0])   # 원본 높이 (h)
    scale = min(max(target_h / h, 1.0), max_scale)
    return cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

def is_blurry(img, thr=80):
    # 입력은 uint8 그레이 이미지라고 가정
    val = cv2.Laplacian(img, cv2.CV_64F).var()
    return val < thr, val

def enhance_plate(img_gray, H=96, W=384,
                  use_bin=False, use_sr=False,
                  small_text_mode=None):

    h, w = img_gray.shape[:2]

    # 0) small_text_mode 자동 결정
    if small_text_mode is None:
        small_text_mode = (min(h, w) < 60)

    # 0-1) 블러 정도 측정
    blurry, blur_score = is_blurry(img_gray, thr=80)

    # 1) (옵션) 슈퍼해상도
    if use_sr:
        # 흐릿하거나 작은 글자면 더 키워보는 것도 가능
        if small_text_mode or blurry:
            target_h = int(1.7 * H)
            max_scale = 3.0
        else:
            target_h = H
            max_scale = 2.5

        img_gray = super_res_adaptive(img_gray,
                                      target_h=target_h,
                                      max_scale=max_scale)

    # 2) 비율 유지 리사이즈 + 패딩
    x = resize_keep_ratio_pad(img_gray, H=H, W=W)

    # 3) 노이즈 억제 (블러면 약하게 or 스킵)
    if blurry:
        # 이미 흐릿한 경우 -> 매우 약하게
        x = cv2.bilateralFilter(x, d=3, sigmaColor=10, sigmaSpace=10)
    else:
        x = cv2.bilateralFilter(
            x, d=5,
            sigmaColor=20 if small_text_mode else 30,
            sigmaSpace=20 if small_text_mode else 30
        )

    # 4) 감마
    x = auto_gamma(x, low=1.25, high=0.75)

    # 5) CLAHE 
    if blurry:
        tiles = (8, 8)
        clip = 1.8
    else:
        tiles = (8, 8) if small_text_mode else (6, 6)
        clip = 1.8 if small_text_mode else 1.5

    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    x = cl.apply(x)

    # 6) 언샤프 마스크
    if blurry:
        blur = cv2.GaussianBlur(x, (0, 0), 1.3)
        x = cv2.addWeighted(x, 1.6, blur, -0.6, 0)
    else:
        blur = cv2.GaussianBlur(x, (0, 0),
                                0.7 if small_text_mode else 0.9)
        x = cv2.addWeighted(
            x,
            1.2 if small_text_mode else 1.4,
            blur,
            -0.2 if small_text_mode else -0.4,
            0
        )

    # 7) 이진화
    if use_bin:
        if blurry:
            win = 21 
            k = 0.24
        else:
            win = 17 if small_text_mode else 31
            k = 0.22 if small_text_mode else 0.25

        x = sauvola_binarize(x, win=win, k=k)

    return x


# --- 이미지 전처리 ---
def to_tensor_01(img):
    # img: uint8 (H,W) or (H,W,3) -> (1,H,W) float32 [0,1]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = img[None, ...]  # (1,H,W)
    return torch.from_numpy(img)

class TextImageDataset(Dataset):
    """
    samples: list of (image_path, label_str)
    img_hw:  (H,W), e.g., (32,128)
    """
    def __init__(self, samples, charset, img_hw=(128, 512), aug=False, max_len=16, use_sr=False, use_bin=False):
        self.items   = samples
        self.charset = charset
        self.H, self.W = img_hw
        self.aug     = aug
        self.max_len = max_len
        self.use_sr  = use_sr   
        self.use_bin = use_bin   

    def _augment(self, img):
        # 가벼운 증강(선택): 밝기/가우시안 노이즈/약한 블러 등
        if random.random() < 0.3:
            alpha = 0.8 + 0.4*random.random()
            img = np.clip(img*alpha, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3,3), 0)
        return img

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지를 불러오지 못했습니다: {path}")

        # 항상 NumPy uint8 보장 (OpenCV 파이프라인용)
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # (선택) 약한 증강
        if self.aug:
            img = self._augment(img)

        # 전처리
        img = enhance_plate(
            img, 
            H=self.H, W=self.W, 
            use_bin=self.use_bin, 
            use_sr=self.use_sr
        )  # 반환: (H, W) uint8

        # 정규화 후 텐서 변환 
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5   # [-1, 1]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)       

        tgt = self.charset.encode(label, self.max_len)
        return img, tgt
    def __len__(self):
        return len(self.items)

def collate_fn(batch):
    # batch: list of (img_tensor(1,H,W), tgt_ids(L))
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)     # (N,1,H,W)
    tgts = torch.stack(tgts, dim=0)     # (N,L)
    return imgs, tgts


import torch.nn.functional as F

@torch.no_grad()
def greedy_decode(model, images, max_len, charset):
    model.eval()
    logits, _ = model(images, tgt_ids=None, max_len=max_len, teacher_forcing=0.0)
    # logits: (N,L,V)
    ids = logits.argmax(-1)  # (N,L)
    texts = [charset.decode(seq.tolist()) for seq in ids]
    return texts

def seq_ce_loss(logits, tgt, pad_idx=0, label_smoothing=0.1, class_weights=None):
    # logits: (N,L,V), tgt: (N,L)
    N, L, V = logits.size()
    logits = logits.reshape(N*L, V)
    tgt = tgt.reshape(N*L)
    return F.cross_entropy(
        logits, tgt, ignore_index=pad_idx, label_smoothing=label_smoothing,  weight=class_weights
    )

def cer(pred, gt):
    # 간단한 edit distance 기반 CER
    import numpy as np
    dp = np.zeros((len(gt)+1, len(pred)+1), dtype=np.int32)
    for i in range(len(gt)+1): dp[i,0] = i
    for j in range(len(pred)+1): dp[0,j] = j
    for i in range(1, len(gt)+1):
        for j in range(1, len(pred)+1):
            cost = 0 if gt[i-1]==pred[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[len(gt), len(pred)] / max(1, len(gt))



from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, device, charset, max_len=16,
                    teacher_forcing=0.5, grad_clip=1.0, scaler=None, class_weights=None):
    model.train()
    total_loss = 0.0
    count = 0

    for images, tgt in loader:
        images = images.to(device)        # (N,1,H,W)
        tgt    = tgt.to(device)           # (N,L)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(scaler is not None)):
            logits, _ = model(images, tgt_ids=tgt, max_len=max_len,
                              teacher_forcing=teacher_forcing)
            loss = seq_ce_loss(logits, tgt, pad_idx=charset.pad, label_smoothing=0.1, class_weights=class_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / max(1, count)

@torch.no_grad()
def validate(model, loader, device, charset, max_len=16,  class_weights=None):
    model.eval()
    total_loss = 0.0
    count = 0
    total_cer = 0.0
    exact = 0

    for images, tgt in loader:
        images = images.to(device)
        tgt    = tgt.to(device)

        logits, _ = model(images, tgt_ids=tgt, max_len=max_len, teacher_forcing=0.0)
        loss = seq_ce_loss(logits, tgt, pad_idx=charset.pad, label_smoothing=0.0, class_weights=class_weights)
        total_loss += loss.item() * images.size(0)
        count += images.size(0)

        # Greedy decode
        ids = logits.argmax(-1)  # (N,L)
        preds = [charset.decode(seq.tolist()) for seq in ids]
        gts   = [charset.decode(seq.tolist()) for seq in tgt]

        for p, g in zip(preds, gts):
            total_cer += cer(p, g)
            if p == g:
                exact += 1

    avg_loss = total_loss / max(1, count)
    avg_cer  = total_cer / max(1, count)
    acc      = exact / max(1, count)
    return {"loss": avg_loss, "cer": avg_cer, "acc": acc}



### 지역이름에 따라 데이터 증강

from collections import Counter
import random, cv2, numpy as np
import torch
from torch.utils.data import Dataset
import re

# ── 정책 설정 ─────────────────────────────────────────────────────────────────
EXEMPT_PREFIXES   = {"경기", "인천", "서울"}                              # 증강 제외 지역
MINORITY_REGIONS  = {"경남","부산","대구","전남","전북","충남","충북","광주","대전","강원"}  # 부족 지역

def get_region_key(lbl: str) -> str:
    """라벨에서 지역 키(앞 2글자)를 뽑되 숫자 시작이면 특수키 반환"""
    s = (lbl or "").strip()
    return s[:2] if s and not s[0].isdigit() else "<DIGIT_FIRST>"

def should_augment(lbl: str) -> bool:
    """조건(숫자 시작, '경기/인천/서울' 시작)이면 False(=증강 제외)"""
    if not lbl: 
        return False
    s = lbl.strip()
    # 1) 숫자로 시작 → 제외
    if s[0].isdigit():
        return False
    # 2) 연속 한글 덩어리의 앞 2글자가 제외 접두사면 제외
    m = re.match(r'[가-힣]{2,}', s)
    if m and m.group(0)[:2] in EXEMPT_PREFIXES:
        return False
    return True

def imread_unicode(path, flag=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)

def down_up(img, s_min=0.5, s_max=0.85):
    h, w = img.shape[:2]
    s = random.uniform(s_min, s_max)
    small = cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

def motion_blur(img, k=5):
    K = np.zeros((k,k), dtype=np.float32); K[k//2, :] = 1.0/k
    return cv2.filter2D(img, -1, K)

def jpeg_noisy(img, q_min=35, q_max=60):
    q = random.randint(q_min, q_max)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def light_geom(img):
    h, w = img.shape[:2]
    M = np.float32([[1, random.uniform(-0.05, 0.05), 0],[0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def thin_stroke(img, p=0.3):
    if random.random() < p:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        return cv2.erode(img, kernel, iterations=1)
    return img

# ── Dataset 본체 ─────────────────────────────────────────────────────────────
class TextImageDataset(Dataset):
    """
    samples: list[(img_path, label)]
    charset: Charset instance (encode/ decode 제공)
    img_hw:  (H,W)  ex) (128,512)
    """
    def __init__(self, samples, charset, img_hw=(128,512), aug=True, max_len=16,
                 use_sr=True, use_bin=False):
        self.items   = samples
        self.charset = charset
        self.H, self.W = img_hw
        self.aug     = aug
        self.max_len = max_len
        self.use_sr  = use_sr
        self.use_bin = use_bin

    # 부족 지역 전용 증강
    def _augment_minor(self, img):
        if random.random() < 0.7: img = down_up(img, 0.5, 0.8)                           # 저해상 재현
        if random.random() < 0.5: img = motion_blur(img, k=random.choice([3,5,7]))       # 모션 블러
        if random.random() < 0.5: img = jpeg_noisy(img, 35, 60)                          # JPEG 아티팩트
        if random.random() < 0.3: img = light_geom(img)                                  # 약한 시어
        if random.random() < 0.4: img = thin_stroke(img, p=1.0)                          # 획 얇게
        return img

    def _augment_major(self, img):
        # 예: 아주 약한 노이즈/블러만
        if random.random() < 0.3: img = jpeg_noisy(img, 60, 80)
        if random.random() < 0.2: img = cv2.GaussianBlur(img, (3,3), random.uniform(0.4,0.8))
        return img

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"read fail: {path}")

        # ── 증강 정책: 두 조건(숫자 시작, '경기/인천/서울')은 제외 ──
        if self.aug and should_augment(label):
            if get_region_key(label) in MINORITY_REGIONS:
                img = self._augment_minor(img)
            # else:
            #     img = self._augment_major(img) 

        # ── 전처리 파이프라인 ──
        img = enhance_plate(img, H=self.H, W=self.W, use_bin=self.use_bin, use_sr=self.use_sr)

        # 정규화 → 텐서
        img = (img.astype(np.float32)/255.0 - 0.5) / 0.5
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)

        tgt = self.charset.encode(label, self.max_len)
        return img, tgt

    def __len__(self):
        return len(self.items)

import os, glob, re
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pymodel import AttnOCR, Charset
from sklearn.model_selection import train_test_split
import numpy as np
import numpy
assert np.__version__ == numpy.__version__,"numpy alias mismatch"
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler

SUFFIX_RE = re.compile(r"-\d+$") 


def base_label_from_stem(stem: str) -> str:
    return SUFFIX_RE.sub("", stem)

def build_charset_from_json_label_folder(label_root="data/label/new"):
    charset = set()
    for jpath in glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True):
        stem = os.path.splitext(os.path.basename(jpath))[0]
        stem = base_label_from_stem(stem)   
        for ch in stem:
            charset.add(ch)
    return "".join(sorted(charset))

IMG_EXTS = (".png", ".jpg", ".jpeg")

def _pick_image(image_root, split, stem):
    base = os.path.join(image_root, split)

    # 1) 동일 이름 우선
    for ext in IMG_EXTS:
        p = os.path.join(base, stem + ext)
        if os.path.isfile(p):
            return os.path.abspath(p)

    # 2) 하위까지 부분일치 탐색
    for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isfile(p):
            name, ext = os.path.splitext(os.path.basename(p))
            if name.startswith(stem) and ext.lower() in IMG_EXTS:
                return os.path.abspath(p)

    return None


def load_lists_from_json_label_folder(label_root="data/label", image_root="data"):
    train_list, val_list = [], []
    for split in ("train", "val"):
        for jpath in glob.glob(os.path.join(label_root, split, "*.json")):
            stem = os.path.splitext(os.path.basename(jpath))[0]
            base_label = SUFFIX_RE.sub("", stem)        
            img_path = _pick_image(image_root, split, stem) or _pick_image(image_root, split, base_label)
            if not img_path: 
                continue
            (train_list if split=="train" else val_list).append((img_path, base_label))
    return sorted(train_list), sorted(val_list)


def quality_score(gray: np.ndarray) -> float:
    # 1) 선명도: 라플라시안 분산 (클수록 sharp)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 2) 대비: 표준편차
    contrast = float(gray.std())
    # 3) 과/저노출 패널티
    hist,_ = np.histogram(gray, bins=256, range=(0,255))
    total = gray.size
    dark = hist[:5].sum()/total
    bright = hist[-5:].sum()/total
    penalty = (dark + bright) * 100  # 가중치
    return sharp + contrast - penalty

def pick_best_per_label(image_dir: str):
    # image_dir: data/train 또는 data/val
    groups = {}
    for p in glob.glob(os.path.join(image_dir, "*")):
        if not os.path.isfile(p): 
            continue
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() not in IMG_EXTS:
            continue
        base = base_label_from_stem(stem)
        gray = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None: 
            continue
        s = quality_score(gray)
        groups.setdefault(base, []).append((s, p))

    best = []
    for base, items in groups.items():
        items.sort(key=lambda x: x[0], reverse=True)   # 점수 내림차순
        best.append(items[0][1])                       # 최고점 한 장
    return best

def build_items(best_paths, label_root_split):
    items = []
    for img_path in best_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        base = base_label_from_stem(stem)
        # 라벨 존재 확인
        j1 = os.path.join(label_root_split, base + ".json")
        j2 = os.path.join(label_root_split, stem + ".json")
        if os.path.isfile(j1) or os.path.isfile(j2):
            items.append((img_path, base))
    return items

train_best = pick_best_per_label("data/train/new")
train_list = build_items(train_best, "data/label/new/train")
train_list, val_list = train_test_split(train_list, test_size=0.2, random_state=42)

print(f"Split: train={len(train_list)} | val={len(val_list)}")



class Charset:
    def __init__(self, charset):
        # 특수 토큰 + 문자셋
        self.idx2ch = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + list(charset)
        self.ch2idx = {ch:i for i,ch in enumerate(self.idx2ch)}
        self.pad, self.sos, self.eos, self.unk = 0,1,2,3

    def encode(self, text, max_len):
        # 문자열을 토큰 ID 시퀀스로 변환
        ids = [self.sos] + [self.ch2idx.get(ch, self.unk) for ch in text][:max_len-2] + [self.eos]
        if len(ids) < max_len:
            ids += [self.pad] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        # 토큰 ID 시퀀스를 문자열로 변환
        out = []
        for i in ids:
            ch = self.idx2ch[i]
            if ch == '<EOS>': break
            if ch not in ['<PAD>','<SOS>','<EOS>','<UNK>']:
                out.append(ch)
        return ''.join(out)
    
charset_str = build_charset_from_json_label_folder("data/label/new")
charset = Charset(charset_str)
print("vocab size:", len(charset.idx2ch))


# 유니코드로 한글 여부 판단
def is_hangul(ch):
    return '\uAC00' <= ch <= '\uD7A3'

# charset.idx2ch 안에서 한글 인덱스만 자동으로 추출
hangul_indices = [
    i for i, ch in enumerate(charset.idx2ch)
    if is_hangul(ch)
]

print("한글 클래스 개수:", len(hangul_indices))

# 클래스 가중치 텐서 생성
class_weights = torch.ones(len(charset.idx2ch), dtype=torch.float32)

# 한글 클래스에만 가중치 2.0 적용 
for idx in hangul_indices:
    class_weights[idx] = 2.0

    

# --- 하이퍼파라미터 ---
IMG_HW = (128, 512)
MAX_LEN = 16
BATCH  = 64          
EPOCHS = 20            
TF_START, TF_END = 0.7, 0.2
LR = 3e-4


SAVE_DIR = "c:/ocr/ocr"
BEST_PATH = os.path.join(SAVE_DIR, "attnocr_best.pt")
LAST_PATH = os.path.join(SAVE_DIR, "attnocr_last.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # --- Dataset ---
    train_ds = TextImageDataset(train_list, charset, IMG_HW, aug=True,  max_len=MAX_LEN, use_sr=True,  use_bin=False)
    val_ds   = TextImageDataset(val_list,   charset, IMG_HW, aug=False, max_len=MAX_LEN, use_sr=True,  use_bin=False)

    
    alpha = 3.0  # 부족 지역을 3배 더 자주 보게
    weights = []
    for (img_path, label) in train_list:
        region = get_region_key(label)
        w = alpha if region in MINORITY_REGIONS and should_augment(label) else 1.0
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.double)


    TIMES = 5  # 에폭당 노출을 5배로
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_list) * TIMES,  
        replacement=True                     
    )
    # ===============================================

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH,
        sampler=train_sampler,         
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    val_dl   = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )


    # --- Model / Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttnOCR(vocab_size=len(charset.idx2ch), img_ch=1, cnn_dim=256, use_bilstm=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    # class_weights = class_weights.to(device)
    cw = class_weights.to(device)

    best_acc = -1.0  # 첫 에폭에도 저장되도록 -1로 시작 (또는 None)

    for epoch in range(1, EPOCHS+1):
        tf_ratio = TF_START + (TF_END - TF_START) * (epoch-1)/(EPOCHS-1)

        train_loss = train_one_epoch(
            model, train_dl, optimizer, device, charset,
            max_len=MAX_LEN, teacher_forcing=tf_ratio, grad_clip=1.0, scaler=scaler, class_weights=cw
        )
        metrics = validate(model, val_dl, device, charset, max_len=MAX_LEN)
        scheduler.step()

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={metrics['loss']:.4f} | "
              f"val_cer={metrics['cer']:.4f} | "
              f"val_acc={metrics['acc']*100:.2f}% | "
              f"TF={tf_ratio:.2f} | LR={scheduler.get_last_lr()[0]:.2e}")

        # 매 에폭 최신 가중치 저장
        torch.save({"model": model.state_dict()}, LAST_PATH)
        print(f"  >> latest saved: {LAST_PATH}")

        # 베스트 갱신 시 저장
        if metrics["acc"] >= best_acc:
            best_acc = metrics["acc"]
            torch.save({
                "model": model.state_dict(),
                "charset": charset.idx2ch,
                "epoch": epoch,
                "acc": best_acc,
            }, BEST_PATH)
            print(f"  >> best updated: {BEST_PATH} (acc={best_acc:.4f})")
        
        print(f"[Epoch {epoch}] seen ~{len(train_ds) * TIMES} samples")

if __name__ == "__main__": 
    main()
    use_sr = True
