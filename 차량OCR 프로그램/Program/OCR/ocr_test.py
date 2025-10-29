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
            raise RuntimeError(f"Failed to read {path}")

        if self.aug:
            img = self._augment(img)

        img = enhance_plate(img, H=self.H, W=self.W, use_bin=self.use_bin, use_sr=self.use_sr)

        img = (img.astype(np.float32)/255.0 - 0.5) / 0.5
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)   # (1,H,W)

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



# loss function
import torch.nn.functional as F

def seq_ce_loss(logits, target, pad_idx=0, label_smoothing=0.0):
    """
    logits: (N, L, V)  | target: (N, L)
    반환값: torch.Tensor (scalar), requires_grad=True
    """
    N, L, V = logits.shape
    logits = logits.reshape(N * L, V)
    target = target.reshape(N * L)

    if label_smoothing > 0:
        n_class = V
        smooth = label_smoothing / (n_class - 1)
        one_hot = torch.full_like(logits, smooth)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        loss_vec = -(one_hot * log_prob).sum(dim=1)     # (N*L,)
    else:
        loss_vec = F.cross_entropy(logits, target, ignore_index=pad_idx, reduction="none")  # (N*L,)

    mask = target != pad_idx
    loss = (loss_vec * mask).sum() / mask.sum().clamp(min=1)  # scalar Tensor
    return loss 


# %%
import torch
from torch.cuda.amp import autocast

@torch.no_grad()
def validate(model, loader, device, charset, max_len=16):
    model.eval()
    total_loss, total_cer, total_acc, n_samples = 0.0, 0.0, 0.0, 0

    for images, tgt in loader:
        images, tgt = images.to(device), tgt.to(device)

        # 1) 손실 계산용: TF=1.0, 정답을 입력으로
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits_tf, _ = model(images, tgt_ids=tgt, max_len=max_len, teacher_forcing=1.0)
            loss = seq_ce_loss(logits_tf, tgt, pad_idx=charset.pad)

        # 2) 예측/평가용: TF=0.0, 입력 없음 (greedy)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits, _ = model(images, tgt_ids=None, max_len=max_len, teacher_forcing=0.0)
        ids = logits.argmax(-1)  # (B, T)

        # 디코드 & 지표
        batch = images.size(0)
        preds = [charset.decode(seq.tolist()) for seq in ids]
        gts   = [charset.decode(seq.tolist()) for seq in tgt]
        for p, g in zip(preds, gts):
            total_cer += char_error_rate(p, g)
            total_acc += int(p == g)

        total_loss += loss.item() * batch
        n_samples  += batch

    return {
        "loss": total_loss / max(1, n_samples),
        "cer":  total_cer  / max(1, n_samples),
        "acc":  total_acc  / max(1, n_samples),
    }

def char_error_rate(pred: str, gt: str) -> float:
    # Levenshtein distance / |gt|
    n, m = len(pred), len(gt)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        pi = pred[i-1]
        for j in range(1, m+1):
            cost = 0 if pi == gt[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m] / max(1, m)


# %%
import os, csv, math, glob, torch, re
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import List, Tuple
from pymodel import AttnOCR, Charset 


# ---------------- JSON 기반 라벨 로더 ----------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SUFFIX_RE = re.compile(r"-\d+$")

def base_label_from_stem(stem: str) -> str:
    return SUFFIX_RE.sub("", stem)


def _pick_image(image_root: str, split: str, stem: str) -> str | None:
    """
    data/{split}/에서 stem과 매칭되는 이미지를 찾는다.
    1) 완전일치 <stem>.<ext>
    2) 부분일치 <stem>-something.<ext> (예: '...6296' ↔ '...6296-3.jpg')
    반환은 절대경로.
    """
    base = os.path.join(image_root, split)

    # 1) 완전일치
    for ext in IMG_EXTS:
        p = os.path.join(base, stem + ext)
        if os.path.isfile(p):
            return os.path.abspath(p)

    # 2) 부분일치 + 재귀
    for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isfile(p):
            name, ext = os.path.splitext(os.path.basename(p))
            if name.startswith(stem) and ext.lower() in [e.lower() for e in IMG_EXTS]:
                return os.path.abspath(p)
    return None


def load_split_from_json(label_root: str = "data/label",
                         image_root: str = "data",
                         split: str = "val") -> List[Tuple[str, str]]:
    items: list[tuple[str,str]] = []
    pattern = os.path.join(label_root, split, "*.json")
    json_paths = glob.glob(pattern)

    missing = 0
    for jpath in json_paths:
        stem = os.path.splitext(os.path.basename(jpath))[0]
        label = base_label_from_stem(stem)  # 접미사 제거 라벨
        # 이미지 매칭: stem → base_label 순서로 시도
        img_path = _pick_image(image_root, split, stem) \
                   or _pick_image(image_root, split, label)
        if img_path is None:
            missing += 1
            continue
        items.append((img_path, label))

    print(f"[LOAD] split={split} json={len(json_paths)}  matched={len(items)}  missing_img={missing}")
    items.sort()
    return items

def build_charset_from_json(label_root: str = "data/label") -> str:
    charset = set()
    for jpath in glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True):
        stem = os.path.splitext(os.path.basename(jpath))[0]
        stem = base_label_from_stem(stem)  
        for ch in stem:
            charset.add(ch)
    return "".join(sorted(charset))

# ---------------- 품질 점수 & 최고품질 1장 선택 ----------------
def quality_score(gray: np.ndarray) -> float:
    """
    간단 품질 점수: 선명도(라플라시안 분산) + 대비(표준편차) - 과/저노출 패널티.
    """
    # 선명도
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 대비
    contrast = float(gray.std())
    # 노출 패널티
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    total = gray.size
    dark = hist[:5].sum() / max(1, total)
    bright = hist[-5:].sum() / max(1, total)
    penalty = (dark + bright) * 100.0
    # (선택) 해상도 가중치 살짝 부여
    w = gray.shape[1]
    return sharp + contrast - penalty + 0.01 * w

def pick_best_per_label(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    (img_path, label) 리스트에서 label별로 그룹핑하여
    품질 점수가 가장 높은 이미지 1장만 남긴다.
    """
    groups: dict[str, list[tuple[float, str]]] = {}
    for path, label in items:
        data = np.fromfile(path, np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        s = quality_score(gray)
        groups.setdefault(label, []).append((s, path))

    best: list[tuple[str, str]] = []
    for label, lst in groups.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        best.append((lst[0][1], label))  # 최고 점수 1장
    print(f"[FILTER] dedup by quality: {len(items)} → {len(best)}")
    return best

# ---------------- CER(Levenshtein) ----------------
def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    return dp[n][m] / n

# ---------------- 메인 테스트 ----------------
def test_eval(
    ckpt_path: str = r"c:/ocr/ocr/attnocr_best.pt",
    label_root: str = r"data/label",        
    image_root: str = r"data",
    split: str = "val",                     
    img_hw=(96, 384),   
    max_len: int = 16,
    batch_size: int = 64,
    num_workers: int = 0,
    use_quality_filter: bool = True,    
    out_csv: str = r"data/preds_val_1027.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 1) 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 2) charset 확보: (A) ckpt에 저장되어 있으면 사용, (B) 없으면 json 기반으로 구성
    if isinstance(ckpt, dict) and "charset" in ckpt and ckpt["charset"]:
        idx2ch = ckpt["charset"]
        base_charset = [ch for ch in idx2ch if ch not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]
        charset = Charset("".join(base_charset))
    else:
        charset_str = build_charset_from_json(label_root)
        charset = Charset(charset_str)

    # 3) 데이터 로드: JSON 라벨에서 해당 split만
    items = load_split_from_json(label_root=label_root, image_root=image_root, split=split)
    if use_quality_filter:
        items = pick_best_per_label(items)
    if len(items) == 0:
        raise RuntimeError(f"'{split}' split에서 사용할 항목이 없습니다. (label_root={label_root})")

    # Train과 동일한 전처리 옵션 사용
    ds = TextImageDataset(items, charset, img_hw, aug=False, max_len=max_len,
                          use_sr=True, use_bin=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=4 if num_workers > 0 else None,
                    collate_fn=collate_fn)

    # 4) 모델 빌드 & 가중치 로드
    model = AttnOCR(vocab_size=len(charset.idx2ch), img_ch=1, cnn_dim=256, use_bilstm=True).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # 5) 추론 루프
    total, sent_ok = 0, 0
    cer_sum, char_count = 0.0, 0
    rows = []

    with torch.no_grad(), autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        for images, tgt in dl:
            images = images.to(device, non_blocking=True)
            tgt    = tgt.to(device, non_blocking=True)

            # 평가용: teacher_forcing=0.0
            logits, _ = model(images, tgt_ids=None, max_len=max_len, teacher_forcing=0.0)
            ids = logits.argmax(-1)

            for b in range(ids.size(0)):
                pred_str = charset.decode(ids[b].tolist())
                gt_str   = charset.decode(tgt[b].tolist())

                this_cer = cer(gt_str, pred_str)
                cer_sum += this_cer * len(gt_str)
                char_count += len(gt_str)
                sent_ok += int(pred_str == gt_str)
                total += 1

                rows.append([items[total-1][0], gt_str, pred_str, f"{this_cer:.4f}"])


    avg_cer = cer_sum / max(1, char_count)
    acc = sent_ok / max(1, total)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "gt", "pred", "cer"])
        w.writerows(rows)

    print(f"[EVAL] split={split} | samples={total} | ACC={acc*100:.2f}% | CER={avg_cer:.4f} | quality_filter={use_quality_filter}")
    print(f"[EVAL] saved: {out_csv}")

if __name__ == "__main__":
    test_eval(
        ckpt_path=r"c:/ocr/ocr/attnocr_best.pt",
        label_root=r"data/label",          
        image_root=r"data",
        split="val",
        img_hw=(96,384),
        max_len=16,
        batch_size=64,
        num_workers=0,
        use_quality_filter=True,
        out_csv=r"data/preds_val_1027.csv"
    )
