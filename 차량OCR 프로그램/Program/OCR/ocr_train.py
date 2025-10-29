# %%
import cv2, numpy, onnxruntime as ort
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)
print("onnxruntime:", ort.__version__)

# %%
import cv2
print(cv2.__version__)
print(cv2.dnn_superres.DnnSuperResImpl_create)

# %%
import os, cv2, math, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def imread_unicode(path, flag=cv2.IMREAD_GRAYSCALE):
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
    mean = cv2.boxFilter(img, ddepth=cv2.CV_32F, ksize=(win,win))
    sqmean = cv2.boxFilter((img*img).astype(np.float32), ddepth=cv2.CV_32F, ksize=(win,win))
    var = np.maximum(sqmean - mean*mean, 0.0)
    std = np.sqrt(var)
    R = 128.0
    thresh = mean*(1 + k*((std/R)-1))
    bin_img = (img.astype(np.float32) > thresh).astype(np.uint8)*255
    return bin_img

_SR = None  # 전역 캐시

def super_res_x2(img_gray):
    """
    단순 업스케일
    저해상 번호판 이미지를 2배 확대 후 부드럽게 보간.
    """
    return cv2.resize(
        img_gray,
        (img_gray.shape[1] * 2, img_gray.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC
    )

def enhance_plate(img_gray, H=64, W=256, use_bin=False, use_sr=False):
    # 0) 필요 시 업스케일 먼저 (원본 기반에서)
    if use_sr:
        img_gray = super_res_x2(img_gray)

    # 1) 비율유지 리사이즈 + 패딩
    x = resize_keep_ratio_pad(img_gray, H=H, W=W)
    # 2) 감마
    x = auto_gamma(x)
    # 3) CLAHE
    x = clahe(x, clip=2.0, tiles=(8,8))
    # 4) 노이즈 제거
    x = denoise(x, method="bilateral")
    # 5) 언샤프
    x = unsharp(x, k=0.8, sigma=1.0)
    # 6) 선택적 이진화
    if use_bin:
        x = sauvola_binarize(x, win=25, k=0.2)
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
    def __init__(self, samples, charset, img_hw=(64, 256), aug=False, max_len=16, use_sr=False, use_bin=False):
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

        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if self.aug:
            img = self._augment(img)

        img = enhance_plate(
            img, 
            H=self.H, W=self.W, 
            use_bin=self.use_bin, 
            use_sr=self.use_sr
        )  

        # 정규화 후 텐서 변환 — 딱 한 번
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


# %%
import torch.nn.functional as F

@torch.no_grad()
def greedy_decode(model, images, max_len, charset):
    model.eval()
    logits, _ = model(images, tgt_ids=None, max_len=max_len, teacher_forcing=0.0)
    # logits: (N,L,V)
    ids = logits.argmax(-1)  # (N,L)
    texts = [charset.decode(seq.tolist()) for seq in ids]
    return texts

def seq_ce_loss(logits, tgt, pad_idx=0, label_smoothing=0.1):
    # logits: (N,L,V), tgt: (N,L)
    N, L, V = logits.size()
    logits = logits.reshape(N*L, V)
    tgt = tgt.reshape(N*L)
    return F.cross_entropy(
        logits, tgt, ignore_index=pad_idx, label_smoothing=label_smoothing
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


# %%
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, device, charset, max_len=16,
                    teacher_forcing=0.5, grad_clip=1.0, scaler=None):
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
            loss = seq_ce_loss(logits, tgt, pad_idx=charset.pad, label_smoothing=0.1)

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
def validate(model, loader, device, charset, max_len=16):
    model.eval()
    total_loss = 0.0
    count = 0
    total_cer = 0.0
    exact = 0

    for images, tgt in loader:
        images = images.to(device)
        tgt    = tgt.to(device)

        logits, _ = model(images, tgt_ids=tgt, max_len=max_len, teacher_forcing=0.0)
        loss = seq_ce_loss(logits, tgt, pad_idx=charset.pad, label_smoothing=0.0)
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


# %%
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

SUFFIX_RE = re.compile(r"-\d+$")  # 끝의 -숫자 제거


def base_label_from_stem(stem: str) -> str:
    return SUFFIX_RE.sub("", stem)

def build_charset_from_json_label_folder(label_root="data/label"):
    charset = set()
    for jpath in glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True):
        stem = os.path.splitext(os.path.basename(jpath))[0]
        stem = base_label_from_stem(stem)   # 접미사 제거 반영
        for ch in stem:
            charset.add(ch)
    return "".join(sorted(charset))

IMG_EXTS = (".png", ".jpg", ".jpeg")

def _pick_image(image_root, split, stem):
    """
    data/{split}/에서 stem과 같은 파일명 이미지를 찾음.
    (예: stem='경기37바6296'이고 이미지가 '경기37바6296-3.jpg'인 경우도 매칭)
    """
    base = os.path.join(image_root, split)

    # 1) 동일 이름 우선
    for ext in IMG_EXTS:
        p = os.path.join(base, stem + ext)
        if os.path.isfile(p):
            return os.path.abspath(p)

    # 2) 하위까지 부분일치 탐색(재귀는 폴더가 평평해도 안전)
    for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isfile(p):
            name, ext = os.path.splitext(os.path.basename(p))
            if name.startswith(stem) and ext.lower() in IMG_EXTS:
                return os.path.abspath(p)

    return None


def load_lists_from_json_label_folder(label_root="data/label", image_root="data"):
    """
    label_root/{train,val}/*.json 을 읽어
    - 라벨: json 파일명(stem) 그대로 (예: '01가3733.json' -> '01가3733')
    - 이미지: data/{train,val}/에서 같은 stem의 이미지 탐색
    """
    train_list, val_list = [], []
    for split in ("train", "val"):
        for jpath in glob.glob(os.path.join(label_root, split, "*.json")):
            stem = os.path.splitext(os.path.basename(jpath))[0]
            base_label = SUFFIX_RE.sub("", stem)        # 접미사 제거
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
    # 3) 과/저노출 패널티: 극단 픽셀 비율
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
    """best_paths: 선택된 이미지 경로 리스트
       label_root_split: data/label/train 또는 data/label/val
       -> (img_path, base_label) 리스트 반환
    """
    items = []
    for img_path in best_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        base = base_label_from_stem(stem)
        # 라벨 존재 확인(선택)
        j1 = os.path.join(label_root_split, base + ".json")
        j2 = os.path.join(label_root_split, stem + ".json")
        if os.path.isfile(j1) or os.path.isfile(j2):
            items.append((img_path, base))
    return items

# 사용 예
train_best = pick_best_per_label("data/train")
# val_best   = pick_best_per_label("data/val")
train_list = build_items(train_best, "data/label/train")
# val_list   = build_items(val_best,   "data/label/val")
# print("after dedup:", len(train_list), len(val_list))
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
    
charset_str = build_charset_from_json_label_folder("data/label")
charset = Charset(charset_str)
print("vocab size:", len(charset.idx2ch))

# --- 하이퍼파라미터 ---
IMG_HW = (96, 384)
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
    # --- Dataset/DataLoader ---
    train_ds = TextImageDataset(train_list, charset, IMG_HW, aug=True,  max_len=MAX_LEN, use_sr=True,  use_bin=False)
    val_ds   = TextImageDataset(val_list,   charset, IMG_HW, aug=False, max_len=MAX_LEN, use_sr=True,  use_bin=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn,
                          drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)

    # --- Model / Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttnOCR(vocab_size=len(charset.idx2ch), img_ch=1, cnn_dim=256, use_bilstm=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_acc = -1.0  

    for epoch in range(1, EPOCHS+1):
        tf_ratio = TF_START + (TF_END - TF_START) * (epoch-1)/(EPOCHS-1)

        train_loss = train_one_epoch(
            model, train_dl, optimizer, device, charset,
            max_len=MAX_LEN, teacher_forcing=tf_ratio, grad_clip=1.0, scaler=scaler
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

        # 베스트 갱신 시 저장 (>= 로 완화)
        if metrics["acc"] >= best_acc:
            best_acc = metrics["acc"]
            torch.save({
                "model": model.state_dict(),
                "charset": charset.idx2ch,
                "epoch": epoch,
                "acc": best_acc,
            }, BEST_PATH)
            print(f"  >> best updated: {BEST_PATH} (acc={best_acc:.4f})")

if __name__ == "__main__":   
    main()
    use_sr = True

# %%
import numpy
print(numpy.__file__)

# %%
# 라벨 json 개수
train_json = glob.glob(os.path.join("data/label","train","*.json"))
val_json   = glob.glob(os.path.join("data/label","val","*.json"))
print("json counts  train/val:", len(train_json), len(val_json))

# 이미지 개수
train_imgs = [p for p in glob.glob(os.path.join("data","train","*.*")) 
              if os.path.splitext(p)[1].lower() in IMG_EXTS]
val_imgs   = [p for p in glob.glob(os.path.join("data","val","*.*")) 
              if os.path.splitext(p)[1].lower() in IMG_EXTS]
print("image counts train/val:", len(train_imgs), len(val_imgs))

# train에서 매칭 실패 예시 몇 개 뽑기
miss = []
for j in train_json:
    stem = os.path.splitext(os.path.basename(j))[0]
    if _pick_image("data","train",stem) is None:
        miss.append(stem)
    if len(miss) >= 5: break
print("train missing examples (first 5):", miss)



