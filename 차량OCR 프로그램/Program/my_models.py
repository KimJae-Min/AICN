"""Streamlit 앱에서 사용하는 번호판 검출 + OCR 추론 모듈.

사하구청 등 배포 환경에서는 학습용 소스를 함께 전달하지 않아도 되도록
추론에 필요한 신경망 구조를 이 파일 안에 직접 포함하였다. 가중치 파일만
교체하면 프로그램이 새로운 모델을 불러온다.
"""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


# ==============================
# MPS 비활성화 (Mac용 GPU fallback 방지)
# ==============================
def disable_mps():
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False


disable_mps()


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class OCRBundle:
    model: torch.nn.Module
    charset: "Charset"
    max_length: int
    img_size: Tuple[int, int]


DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하서울부산대구인천광주대전울산경기강원충북충남전북전남경북경남제주세종"
DEFAULT_DET_IMG_SIZE = 512
DEFAULT_DET_CONF = 0.3
DEFAULT_DET_IOU = 0.4
LP_STRIDE = 8


# ==============================
# 번호판 검출 모델 구조 (LP_YOLO_Fuse)
# ==============================
def autopad(k: int, p: Optional[int] = None) -> int:
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise-Separable convolution."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, act: bool = True) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, autopad(k), groups=c_in, bias=False)
        self.dw_bn = nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.03)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class DSBottleneck(nn.Module):
    def __init__(self, c: int, k: int = 3, expansion: float = 0.5) -> None:
        super().__init__()
        ch = max(8, int(c * expansion))
        self.cv1 = ConvBNAct(c, ch, k=1, s=1)
        self.dw = DSConv(ch, ch, k=k, s=1)
        self.cv2 = ConvBNAct(ch, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.dw(self.cv1(x)))


class C3k2Lite(nn.Module):
    def __init__(self, c: int, n: int = 1, use_k5: bool = True) -> None:
        super().__init__()
        layers = []
        for i in range(n):
            k = 5 if (use_k5 and i % 2 == 1) else 3
            layers.append(DSBottleneck(c, k=k, expansion=0.5))
        self.m = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class SPPF_Tiny(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(c, c, k=1, s=1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.cv2 = ConvBNAct(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.pool(self.cv1(x)))


class PSALite(nn.Module):
    def __init__(self, k: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, autopad(k), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attention


class BackboneLite(nn.Module):
    def __init__(self, in_ch: int = 3, c1: int = 32, c2: int = 64, c3: int = 96) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_ch, c1, k=3, s=2)
        self.stage1 = C3k2Lite(c1, n=1)
        self.down2 = DSConv(c1, c2, k=3, s=2)
        self.stage2 = C3k2Lite(c2, n=2)
        self.down3 = DSConv(c2, c3, k=3, s=2)
        self.stage3 = C3k2Lite(c3, n=3)
        self.sppf = SPPF_Tiny(c3)
        self.out_c = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        return self.sppf(x)


class NeckPANLite(nn.Module):
    def __init__(self, c_in: int = 96, c_mid: int = 32, use_psa: bool = True) -> None:
        super().__init__()
        self.lateral = ConvBNAct(c_in, c_mid, k=1, s=1)
        self.merge = C3k2Lite(c_mid, n=1)
        self.psa = PSALite(k=7) if use_psa else nn.Identity()
        self.out_c = c_mid

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        y = self.lateral(p3)
        y = self.merge(y)
        return self.psa(y)


class DecoupledHead(nn.Module):
    def __init__(self, c_in: int = 32, num_classes: int = 1, use_dfl: bool = False, bins: int = 8) -> None:
        super().__init__()
        self.use_dfl = use_dfl
        self.bins = bins

        def make_set():
            stem = ConvBNAct(c_in, c_in, k=3, s=1)
            reg = nn.Sequential(
                ConvBNAct(c_in, c_in, k=3, s=1),
                nn.Conv2d(c_in, (4 if not use_dfl else 4 * bins), 1),
            )
            obj = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, 1, 1))
            cls = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, num_classes, 1))
            return stem, reg, obj, cls

        self.stem_m, self.reg_m, self.obj_m, self.cls_m = make_set()
        self.stem_o, self.reg_o, self.obj_o, self.cls_o = make_set()

    def forward(self, x: torch.Tensor):
        xm = self.stem_m(x)
        reg_m = self.reg_m(xm)
        obj_m = self.obj_m(xm)
        cls_m = self.cls_m(xm)

        xo = self.stem_o(x)
        reg_o = self.reg_o(xo)
        obj_o = self.obj_o(xo)
        cls_o = self.cls_o(xo)
        return {"o2m": (reg_m, obj_m, cls_m), "o2o": (reg_o, obj_o, cls_o)}


class LP_YOLO_Fuse(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 1, use_psa: bool = True, use_dfl: bool = False, bins: int = 8) -> None:
        super().__init__()
        self.backbone = BackboneLite(in_ch=in_ch)
        self.neck = NeckPANLite(c_in=self.backbone.out_c, c_mid=32, use_psa=use_psa)
        self.head = DecoupledHead(c_in=self.neck.out_c, num_classes=num_classes, use_dfl=use_dfl, bins=bins)
        self.stride = LP_STRIDE

    def forward(self, x: torch.Tensor):
        p3 = self.backbone(x)
        f = self.neck(p3)
        return self.head(f)


def dist2bbox_ltbr(ltrb: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ==============================
# OCR 모델 구조 (AttnOCR)
# ==============================
class Charset:
    def __init__(self, charset: str) -> None:
        self.idx2ch = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(charset)
        self.ch2idx = {ch: i for i, ch in enumerate(self.idx2ch)}
        self.pad, self.sos, self.eos, self.unk = 0, 1, 2, 3

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.sos]
        ids.extend(self.ch2idx.get(ch, self.unk) for ch in text[: max_len - 2])
        ids.append(self.eos)
        if len(ids) < max_len:
            ids.extend([self.pad] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        result = []
        for idx in ids:
            ch = self.idx2ch[idx]
            if ch == "<EOS>":
                break
            if ch not in {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}:
                result.append(ch)
        return "".join(result)


class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)
        n, c, h, w = f.shape
        f = F.adaptive_avg_pool2d(f, (1, w)).squeeze(2)
        f = f.permute(2, 0, 1)
        return f


class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256, num_layers: int = 2, bidir: bool = True) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hid, num_layers=num_layers, bidirectional=bidir)
        self.out_dim = hid * (2 if bidir else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, dec_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N, E = enc_out.size()
        e_proj = self.W_e(enc_out)
        d_proj = self.W_d(dec_h).unsqueeze(0)
        scores = self.v(torch.tanh(e_proj + d_proj)).squeeze(-1)
        alpha = torch.softmax(scores, dim=0)
        ctx = (alpha.unsqueeze(-1) * enc_out).sum(0)
        return ctx, alpha


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size: int, enc_dim: int = 256, dec_dim: int = 256, emb_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.attn = AdditiveAttention(enc_dim, dec_dim)
        self.fc = nn.Linear(dec_dim + enc_dim, vocab_size)

    def forward(
        self,
        enc_out: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        max_len: int = 16,
        teacher_forcing: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N, _ = enc_out.size()
        device = enc_out.device
        h = torch.zeros(N, self.rnn.hidden_size, device=device)
        c = torch.zeros_like(h)
        y = torch.full((N,), fill_value=1, dtype=torch.long, device=device)
        logits: List[torch.Tensor] = []
        atts: List[torch.Tensor] = []
        for t in range(max_len):
            emb = self.embedding(y)
            ctx, alpha = self.attn(enc_out, h)
            rnn_in = torch.cat([emb, ctx], dim=-1)
            h, c = self.rnn(rnn_in, (h, c))
            out = torch.cat([h, ctx], dim=-1)
            logit = self.fc(out)
            logits.append(logit)
            atts.append(alpha.permute(1, 0))
            if tgt is not None and torch.rand(1).item() < teacher_forcing:
                y = tgt[:, t]
            else:
                y = logit.argmax(dim=-1)
        logits_tensor = torch.stack(logits, dim=1)
        atts_tensor = torch.stack(atts, dim=1)
        return logits_tensor, atts_tensor


class AttnOCR(nn.Module):
    def __init__(self, vocab_size: int, img_ch: int = 1, cnn_dim: int = 256, use_bilstm: bool = True) -> None:
        super().__init__()
        self.cnn = CNNEncoder(in_ch=img_ch, out_ch=cnn_dim)
        if use_bilstm:
            self.seq = SeqEncoder(cnn_dim, hid=256, bidir=True)
            enc_dim = self.seq.out_dim
        else:
            self.seq = nn.Identity()
            enc_dim = cnn_dim
        self.dec = AttnDecoder(vocab_size=vocab_size, enc_dim=enc_dim, dec_dim=256, emb_dim=128)

    def forward(
        self,
        x: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        max_len: int = 16,
        teacher_forcing: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.cnn(x)
        enc = self.seq(enc)
        return self.dec(enc, tgt=tgt_ids, max_len=max_len, teacher_forcing=teacher_forcing)


# ==============================
# 가중치 경로 로딩
# ==============================
def _load_weight_config() -> dict:
    config_path = BASE_DIR / "weights_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_weight_path(env_key: str, config_key: str, config: dict) -> Path:
    env_path = os.environ.get(env_key)
    if env_path:
        candidate = Path(env_path)
    else:
        rel = config.get(config_key)
        if not rel:
            raise FileNotFoundError(f"{config_key} 가중치 경로가 설정되지 않았습니다.")
        candidate = (BASE_DIR / rel).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {candidate}")
    return candidate


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "weights"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def _extract_meta(ckpt: dict) -> dict:
    meta: dict = {}
    if isinstance(ckpt, dict):
        for key in ("meta", "config"):
            val = ckpt.get(key)
            if isinstance(val, dict):
                meta.update(val)
        for key in ("charset", "max_len", "img_size", "img_hw", "cnn_dim", "use_bilstm"):
            if key in ckpt and ckpt[key] is not None:
                meta[key] = ckpt[key]
    return meta


def _load_plate_detector(weight_path: Path, device: torch.device) -> torch.nn.Module:
    model = LP_YOLO_Fuse().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.dist2bbox = dist2bbox_ltbr
    return model


def _load_ocr_bundle(weight_path: Path, device: torch.device) -> OCRBundle:
    ckpt = torch.load(weight_path, map_location=device)
    meta = _extract_meta(ckpt)

    charset_str = meta.get("charset")
    if isinstance(charset_str, (list, tuple)):
        charset_str = "".join(charset_str)
    if not charset_str:
        charset_str = DEFAULT_CHARSET

    max_len = int(meta.get("max_len", 16))
    img_hw = meta.get("img_size") or meta.get("img_hw") or (32, 128)
    if isinstance(img_hw, (list, tuple)) and len(img_hw) == 2:
        img_hw = (int(img_hw[0]), int(img_hw[1]))
    else:
        img_hw = (32, 128)
    cnn_dim = int(meta.get("cnn_dim", 256))
    use_bilstm = bool(meta.get("use_bilstm", True))

    charset = Charset(charset_str)
    vocab_size = len(charset.idx2ch)
    model = AttnOCR(vocab_size=vocab_size, img_ch=1, cnn_dim=cnn_dim, use_bilstm=use_bilstm)
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return OCRBundle(model=model, charset=charset, max_length=max_len, img_size=(img_hw[0], img_hw[1]))


# ==============================
# 추론 유틸리티
# ==============================
def _build_grid(height: int, width: int, stride: int, device: torch.device):
    ys = torch.arange(height, device=device) + 0.5
    xs = torch.arange(width, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    return cx.reshape(-1) * stride, cy.reshape(-1) * stride


def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    try:
        from torchvision.ops import nms

        return nms(boxes, scores, iou_threshold)
    except Exception:
        order = scores.argsort(descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            ious = bbox_iou_xyxy(boxes[rest], boxes[i].unsqueeze(0)).squeeze(-1)
            order = rest[ious <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _run_plate_detection(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    img_size: int = DEFAULT_DET_IMG_SIZE,
    conf_thr: float = DEFAULT_DET_CONF,
    iou_thr: float = DEFAULT_DET_IOU,
) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    orig_h, orig_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        reg_o, obj_o, cls_o = outputs["o2o"]

    B, _, H, W = obj_o.shape
    stride = int(getattr(model, "stride", LP_STRIDE))
    cx, cy = _build_grid(H, W, stride, tensor.device)

    if reg_o.shape[1] == 4:
        reg = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(min=0)
    else:
        bins = reg_o.shape[1] // 4
        reg_logits = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4, bins)
        prob = torch.softmax(reg_logits, dim=-1)
        idx = torch.arange(bins, device=reg_logits.device, dtype=reg_logits.dtype)
        reg = (prob * idx).sum(-1)

    boxes = dist2bbox_ltbr(reg, cx.view(-1), cy.view(-1)).view(B, -1, 4)
    scores = torch.sigmoid(obj_o).view(B, -1) * torch.sigmoid(cls_o).view(B, -1)

    boxes = boxes[0]
    scores = scores[0]
    valid = scores > conf_thr
    boxes = boxes[valid]
    scores = scores[valid]

    keep = _nms_indices(boxes, scores, iou_thr)
    if keep.numel() > 0:
        boxes = boxes[keep]
        scores = scores[keep]
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        scores = scores[order]
    else:
        boxes = boxes.new_zeros((0, 4))
        scores = scores.new_zeros((0,))

    if boxes.numel() == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scale_x = orig_w / float(img_size)
    scale_y = orig_h / float(img_size)
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes.cpu().numpy(), scores.cpu().numpy()


def _clean_text(text: str) -> str:
    return text.replace(",", "").replace("-", "").strip()


def _run_ocr_on_crop(crop_rgb: np.ndarray, bundle: OCRBundle) -> str:
    model = bundle.model
    device = next(model.parameters()).device
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, w = bundle.img_size
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(tensor, tgt_ids=None, max_len=bundle.max_length, teacher_forcing=0.0)
    ids = logits.argmax(-1)[0].tolist()
    text = bundle.charset.decode(ids)
    return _clean_text(text)


def _save_plate_crop(crop_rgb: np.ndarray) -> str:
    crops_dir = BASE_DIR / "plate_crops"
    crops_dir.mkdir(exist_ok=True)
    file_path = crops_dir / f"plate_{uuid.uuid4().hex}.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    return str(file_path)


# ==============================
# 공개 API
# ==============================
def load_models():
    config = _load_weight_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_weight = _resolve_weight_path("PLATE_DET_WEIGHTS", "plate_detector", config)
    ocr_weight = _resolve_weight_path("PLATE_OCR_WEIGHTS", "ocr", config)

    plate_detector = _load_plate_detector(det_weight, device)
    ocr_bundle = _load_ocr_bundle(ocr_weight, device)
    return plate_detector, ocr_bundle


def detect_car_plate(img_path: str, plate_detector: torch.nn.Module, ocr_bundle: OCRBundle):
    """번호판 인식 + 번호판 크롭 이미지 경로를 리턴"""
    try:
        im_pil = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return ["인식 불가 (이미지 파일 아님)"], []

    img = np.array(im_pil)
    texts: List[str] = []
    plate_imgs: List[str] = []

    boxes, _ = _run_plate_detection(img, plate_detector)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(img.shape[1] - 1, x1))
        y1 = max(0, min(img.shape[0] - 1, y1))
        x2 = max(0, min(img.shape[1], x2))
        y2 = max(0, min(img.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2].copy()
        crop_path = _save_plate_crop(crop)
        plate_imgs.append(crop_path)
        text = _run_ocr_on_crop(crop, ocr_bundle)
        if text:
            texts.append(text)

    if not texts:
        fallback = _run_ocr_on_crop(img, ocr_bundle)
        texts = [fallback] if fallback else ["인식 실패"]

    return texts, plate_imgs
