# %%
import cv2, numpy, onnxruntime as ort
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)
print("onnxruntime:", ort.__version__)

# %%
import os, cv2, math, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, samples, charset, img_hw=(32,128), aug=False, max_len=16):
        self.samples = samples
        self.charset = charset
        self.H, self.W = img_hw
        self.aug = aug
        self.max_len = max_len

    def _augment(self, img):
        if random.random() < 0.3:
            alpha = 0.8 + 0.4*random.random()
            img = np.clip(img*alpha, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3,3), 0)
        return img

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.aug:
            img = self._augment(img)
        tensor = to_tensor_01(img)  # (1,H,W)
        tgt_ids = self.charset.encode(text, max_len=self.max_len)  # (L,)
        return tensor, tgt_ids

    def __len__(self):
        return len(self.samples)

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
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pymodel import AttnOCR, Charset

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
    
def load_lists_from_labels(labels_txt, base_dir="data"):
       train_list, val_list = [], []
       with open(labels_txt, "r", encoding="utf-8") as f:
           for line in f:
               line = line.strip()
               if not line: 
                   continue
               rel_path, label = line.split(None, 1)  # 공백으로 경로/라벨 분리
               img_path = os.path.join(base_dir, rel_path.replace("/", os.sep))
               # 확장자 확인
               if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                   img_path += ".png"
               if rel_path.startswith("train/"):
                   train_list.append((img_path, label))
               elif rel_path.startswith("val/"):
                   val_list.append((img_path, label))
       return train_list, val_list

train_list, val_list = load_lists_from_labels("data/labels.txt", base_dir="data")

def build_charset_from_labels(labels_txt):
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

charset_str = build_charset_from_labels("data/labels.txt")
charset = Charset(charset_str)
print("vocab size:", len(charset.idx2ch))


# --- 하이퍼파라미터 ---
IMG_HW = (32,128)
MAX_LEN = 16
BATCH  = 2            
EPOCHS = 3            
TF_START, TF_END = 0.7, 0.2
LR = 3e-4

SAVE_DIR = "c:/ocr/ocr"
BEST_PATH = os.path.join(SAVE_DIR, "attnocr_best.pt")
LAST_PATH = os.path.join(SAVE_DIR, "attnocr_last.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # --- Dataset/DataLoader ---
    train_ds = TextImageDataset(train_list, charset, IMG_HW, aug=True,  max_len=MAX_LEN)
    val_ds   = TextImageDataset(val_list,   charset, IMG_HW, aug=False, max_len=MAX_LEN)

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

        # 에폭 최신 가중치 저장
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

if __name__ == "__main__":  
    main()



