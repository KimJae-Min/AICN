# %%
import torch
import torch.nn as nn

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on", device)
if device.type == "cpu":
    print("GPU가 감지되지 않아 CPU에서 학습 중입니다.")


# ## 1. Text Detecion

# %%
import torch, torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large

class FPN(nn.Module):
    def __init__(self, in_chs=(40, 112, 160), out_ch=128):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_chs])

    def forward(self, feats):  # feats: [C3,C4,C5]
        c3, c4, c5 = feats
        p5 = self.lats[2](c5)
        p4 = self.lats[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lats[0](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p5, p4, p3 = self.smooth[2](p5), self.smooth[1](p4), self.smooth[0](p3)
        return p3  # DB는 보통 최상단/병합출력 1개만 사용(필요시 concat)

class DBHead(nn.Module):
    def __init__(self, in_ch, k=50):
        super().__init__()
        self.bin = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())
        self.thr = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())
        self.k = k

    def forward(self, x):
        p = self.bin(x)                      # probability map
        t = self.thr(x)                      # threshold map
        b = 1.0 / (1.0 + torch.exp(-self.k * (p - t)))  # binary map
        return {'p': p, 't': t, 'b': b}

class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        mbv3 = mobilenet_v3_large(weights=None)
        self.stage2 = nn.Sequential(mbv3.features[0:6])   # C3-ish
        self.stage3 = nn.Sequential(mbv3.features[6:10])  # C4-ish
        self.stage4 = nn.Sequential(mbv3.features[10:])   # C5-ish
        self.fpn = FPN(in_chs=(40,112,160), out_ch=128)
        self.head = DBHead(128)

    def forward(self, x):
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        f  = self.fpn([c3,c4,c5])
        out = self.head(f)
        return out


# ## 2. Text Recognition

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) 토크나이저 (예: 숫자 + 한글 일부 + 특수)
# -----------------------------
class Charset:
    def __init__(self, charset):
        self.idx2ch = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + list(charset)
        self.ch2idx = {ch:i for i,ch in enumerate(self.idx2ch)}
        self.pad, self.sos, self.eos, self.unk = 0,1,2,3

    def encode(self, text, max_len):
        ids = [self.sos] + [self.ch2idx.get(ch, self.unk) for ch in text][:max_len-2] + [self.eos]
        if len(ids) < max_len:
            ids += [self.pad] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        out = []
        for i in ids:
            ch = self.idx2ch[i]
            if ch == '<EOS>': break
            if ch not in ['<PAD>','<SOS>','<EOS>','<UNK>']:
                out.append(ch)
        return ''.join(out)


# -----------------------------
# 2) 인코더: CNN → (N, C, H, W) → (T, N, D)
# -----------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2),  # H/2, W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2,2)),# H/4, W/4
            nn.Conv2d(128, out_ch, 3,1,1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
        # BiLSTM로 넘길 임베딩 차원
        self.out_ch = out_ch

    def forward(self, x):
        # x: (N,1, H, W) => (N,C,H',W')
        f = self.body(x)
        N, C, H, W = f.shape
        # 높이를 collapse/ 폭 방향을 time으로 (H를 평균/맥스 풀링)
        f = F.adaptive_avg_pool2d(f, (1, W)).squeeze(2)  # (N, C, W)
        f = f.permute(2, 0, 1)  # (T=W, N, C)
        return f  # (T, N, D=C)

# -----------------------------
# 3) 어텐션 디코더 (Additive/Bahdanau)
# -----------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, dec_h):
        # enc_out: (T, N, enc_dim), dec_h: (N, dec_dim)
        T, N, E = enc_out.size()
        e_proj = self.W_e(enc_out)              # (T,N,A)
        d_proj = self.W_d(dec_h).unsqueeze(0)   # (1,N,A)
        scores = self.v(torch.tanh(e_proj + d_proj)).squeeze(-1)  # (T,N)
        alpha  = F.softmax(scores, dim=0)       # time softmax
        ctx    = (alpha.unsqueeze(-1) * enc_out).sum(0)  # (N,E)
        return ctx, alpha  # context, attention weights

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, enc_dim=256, dec_dim=256, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.attn = AdditiveAttention(enc_dim, dec_dim)
        self.fc   = nn.Linear(dec_dim + enc_dim, vocab_size)

    def forward(self, enc_out, tgt=None, max_len=16, teacher_forcing=0.5):
        T, N, E = enc_out.size()
        device = enc_out.device

        h = torch.zeros(N, self.rnn.hidden_size, device=device)
        c = torch.zeros_like(h)

        # 첫 입력: <SOS>
        y = torch.full((N,), fill_value=1, dtype=torch.long, device=device)  # <SOS>
        logits = []
        atts   = []

        for t in range(max_len):
            emb = self.embedding(y)  # (N, emb_dim)
            ctx, alpha = self.attn(enc_out, h)  # (N,E), (T,N)
            rnn_in = torch.cat([emb, ctx], dim=-1)
            h, c = self.rnn(rnn_in, (h,c))
            out = torch.cat([h, ctx], dim=-1)
            logit = self.fc(out)  # (N, vocab)
            logits.append(logit)
            atts.append(alpha.permute(1,0))  # (N,T)

            # 다음 입력 토큰
            if (tgt is not None) and (torch.rand(1).item() < teacher_forcing):
                y = tgt[:, t]  # teacher forcing
            else:
                y = logit.argmax(dim=-1)

        logits = torch.stack(logits, dim=1)  # (N, max_len, vocab)
        atts   = torch.stack(atts, dim=1)    # (N, max_len, T)
        return logits, atts

# -----------------------------
# 4) 전체 인식기: Encoder + (BiLSTM) + Decoder
# -----------------------------
class SeqEncoder(nn.Module):
    def __init__(self, in_dim, hid=256, num_layers=2, bidir=True):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hid, num_layers=num_layers, bidirectional=bidir)
        self.out_dim = hid * (2 if bidir else 1)

    def forward(self, x):  # x: (T,N,D)
        out,_ = self.rnn(x)  # (T,N,out_dim)
        return out

class AttnOCR(nn.Module):
    def __init__(self, vocab_size, img_ch=1, cnn_dim=256, use_bilstm=True):
        super().__init__()
        self.cnn = CNNEncoder(in_ch=img_ch, out_ch=cnn_dim)
        if use_bilstm:
            self.seq = SeqEncoder(cnn_dim, hid=256, bidir=True)
            enc_dim = self.seq.out_dim
        else:
            self.seq = nn.Identity()
            enc_dim = cnn_dim
        self.dec = AttnDecoder(vocab_size=vocab_size, enc_dim=enc_dim, dec_dim=256, emb_dim=128)

    def forward(self, x, tgt_ids=None, max_len=16, teacher_forcing=0.5):
        # x: (N,1,H,W), tgt_ids: (N, max_len) - (미리 <EOS>까지 포함)
        enc = self.cnn(x)            # (T,N,D)
        enc = self.seq(enc)          # (T,N,E)
        logits, atts = self.dec(enc, tgt=tgt_ids, max_len=max_len, teacher_forcing=teacher_forcing)
        return logits, atts



