# %% [markdown]
# ### model

# %%
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TextImageDataset(Dataset):
    def __init__(self, samples, charset, img_hw=(32,128), aug=False, max_len=32):
        """
        samples: [(img_path, text), ...]
        charset: Charset 객체
        img_hw: (H, W) 리사이즈 크기
        aug: 데이터 증강 여부
        """
        self.samples = samples
        self.charset = charset
        self.H, self.W = img_hw
        self.aug = aug
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {path} | exists={os.path.exists(path)}")

        # 리사이즈
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # 0~1 정규화 + (C,H,W) 변환
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

        # 텍스트-> 정수 시퀀스로 변환
        tgt = self.charset.encode(text, self.max_len)
        return img, tgt


def collate_fn(batch):
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)     # (N,1,H,W)
    tgts = torch.stack(tgts, 0)     # (N,L)
    return imgs, tgts


# %% [markdown]
# ### loss function

# %%
import torch.nn.functional as F

def seq_ce_loss(logits, target, pad_idx=0, label_smoothing=0.0):

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


# %% [markdown]
# ### validation function

# %%
import torch

@torch.no_grad()
def validate(model, loader, device, charset, max_len=16):
    model.eval()
    total_loss, total_cer, total_acc, count = 0.0, 0.0, 0.0, 0

    for images, tgt in loader:
        images, tgt = images.to(device), tgt.to(device)

        # forward (teacher forcing off for validation)
        logits, _ = model(images, tgt_ids=tgt, max_len=max_len, teacher_forcing=0.0)
        loss = seq_ce_loss(logits, tgt, pad_idx=charset.pad)

        # 예측값
        ids = logits.argmax(-1)  # (N,L)
        preds = [charset.decode(seq.tolist()) for seq in ids]
        gts   = [charset.decode(seq.tolist()) for seq in tgt]

        # CER/Acc 계산
        for p, g in zip(preds, gts):
            cer = char_error_rate(p, g)
            acc = int(p == g)
            total_cer += cer
            total_acc += acc
            count += 1

        total_loss += loss.item() * images.size(0)

    metrics = {
        "loss": total_loss / max(1, count),
        "cer":  total_cer / max(1, count),
        "acc":  total_acc / max(1, count),
    }
    return metrics


def char_error_rate(pred, gt):
    """CER 계산 (Levenshtein distance 기반)"""
    # DP 편집거리
    dp = [[0] * (len(gt)+1) for _ in range(len(pred)+1)]
    for i in range(len(pred)+1): dp[i][0] = i
    for j in range(len(gt)+1):   dp[0][j] = j

    for i in range(1, len(pred)+1):
        for j in range(1, len(gt)+1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)

    dist = dp[len(pred)][len(gt)]
    return dist / max(1, len(gt))


# %%
import os, csv, math, torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import List, Tuple

from pymodel import AttnOCR, Charset 


# ---------  라벨/목록 로더 ----------
def load_split_from_labels(labels_txt: str, base_dir="data", split="val") -> List[Tuple[str, str]]:
    items = []
    with open(labels_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split(None, 1)
            if not rel_path.startswith(split + "/"):
                continue
            img_path = os.path.join(base_dir, rel_path.replace("/", os.sep))
            items.append((img_path, label))
    return items

def build_charset_from_labels(labels_txt: str) -> str:
    charset = set()
    with open(labels_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(None, 1)
            for ch in label:
                charset.add(ch)
    return "".join(sorted(charset))

# ---------  CER(Levenshtein) ----------
def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 삭제
                dp[i][j-1] + 1,      # 삽입
                dp[i-1][j-1] + cost  # 치환
            )
    if n == 0:
        return 0.0 if m == 0 else 1.0
    return dp[n][m] / n

# --------- 메인 테스트 ---------
def test_eval(
    ckpt_path: str = r"c:/ocr/ocr/attnocr_best.pt",
    labels_path: str = r"data/labels.txt",
    split: str = "val",                   
    img_hw=(32, 128),
    max_len: int = 16,
    batch_size: int = 64,
    num_workers: int = 0,
    out_csv: str = r"data/preds_val.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 2) charset 확보
    if isinstance(ckpt, dict) and "charset" in ckpt:
        idx2ch = ckpt["charset"]
        charset = Charset("".join([ch for ch in idx2ch if ch not in ['<PAD>','<SOS>','<EOS>','<UNK>']]))
    else:
        charset_str = build_charset_from_labels(labels_path)
        charset = Charset(charset_str)

    # 3) 데이터 로드 
    items = load_split_from_labels(labels_path, base_dir="data", split=split)
    if len(items) == 0:
        raise RuntimeError(f"'{split}' 항목이 labels.txt에 없습니다.")
    ds = TextImageDataset(items, charset, img_hw, aug=False, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    # 4) 모델 빌드 & 가중치 로드
    model = AttnOCR(vocab_size=len(charset.idx2ch), img_ch=1, cnn_dim=256, use_bilstm=True).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # 5) 추론 루프
    total, sent_ok = 0, 0
    cer_sum, char_count = 0.0, 0
    rows = []  # CSV 저장용

    with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
        for images, tgt in dl:
            images = images.to(device)
            logits, _ = model(images, tgt_ids=None, max_len=max_len, teacher_forcing=0.0)
            ids = logits.argmax(-1)  # (B, T)
            # 배치 단위 디코딩/평가
            for b in range(ids.size(0)):
                pred_str = charset.decode(ids[b].tolist())
                gt_str   = charset.decode(tgt[b].tolist())

                this_cer = cer(gt_str, pred_str)
                cer_sum += this_cer * len(gt_str) 
                char_count += len(gt_str)
                sent_ok += int(pred_str == gt_str)
                total += 1

                rows.append([items[total-1][0], gt_str, pred_str, f"{this_cer:.4f}"])

    # 6) 지표 집계
    avg_cer = cer_sum / max(1, char_count)
    acc = sent_ok / max(1, total)

    # 7) 결과 저장
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "gt", "pred", "cer"])
        w.writerows(rows)

    print(f"[EVAL] split={split} | samples={total} | ACC={acc*100:.2f}% | CER={avg_cer:.4f}")
    print(f"[EVAL] saved: {out_csv}")

if __name__ == "__main__":
    test_eval(
        ckpt_path=r"c:/ocr/ocr/attnocr_best.pt",  # 학습 때 저장한 베스트 경로
        labels_path=r"data/labels.txt",
        split="val",                               
        img_hw=(32, 128),
        max_len=16,
        batch_size=64,
        num_workers=0,
        out_csv=r"data/preds_val.csv"
    )

