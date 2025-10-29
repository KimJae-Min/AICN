 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/차량OCR 프로그램/Program/my_models.py b/차량OCR 프로그램/Program/my_models.py
index 78e69544f852650a5df63a936547f2dc21d8abea..3bc38e5213ac146f299ca51574ac7cf8269221a5 100644
--- a/차량OCR 프로그램/Program/my_models.py	
+++ b/차량OCR 프로그램/Program/my_models.py	
@@ -1,109 +1,646 @@
-# models.py
-import torch
-import numpy as np
+import json
+import os
+import uuid
+from dataclasses import dataclass
+from pathlib import Path
+from typing import List, Optional, Tuple
+
 import cv2
-from PIL import Image, ExifTags, UnidentifiedImageError
-from paddleocr import PaddleOCR
+import numpy as np
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+from PIL import Image, UnidentifiedImageError
+
 
 # ==============================
 # MPS 비활성화 (Mac용 GPU fallback 방지)
 # ==============================
 def disable_mps():
     if hasattr(torch.backends, "mps"):
         torch.backends.mps.is_available = lambda: False
         torch.backends.mps.is_built = lambda: False
 
+
 disable_mps()
 
+
+BASE_DIR = Path(__file__).resolve().parent
+
+# ---------------------------------------------------------------------------
+# 커스텀 모델 연동 개요
+# ---------------------------------------------------------------------------
+# 1. 필요한 추론용 신경망 구조(LP_YOLO_Fuse, AttnOCR)를 이 파일 안에 직접 정의해
+#    배포 시 별도의 학습 코드 폴더를 묶지 않아도 된다.
+# 2. weights_config.json 혹은 환경변수로 전달된 경로에서 가중치 파일을 찾는다.
+#    (사하구청 배포 시에는 가중치 파일만 교체하면 되고, 코드 수정이 필요 없다.)
+# 3. 번호판 검출 → 크롭 저장 → OCR 추론 순으로 Streamlit 앱에서 재사용할 수 있는
+#    헬퍼 함수들을 제공한다.
+#
+# 이런 흐름을 한눈에 파악할 수 있도록 상단에 정리해 두었다.
+
+
+# ---------------------------------------------------------------------------
+# 번호판 검출 모델 (LP_YOLO_Fuse) 정의
+# ---------------------------------------------------------------------------
+
+
+def _autopad(k: int, p: Optional[int] = None) -> int:
+    return k // 2 if p is None else p
+
+
+class ConvBNAct(nn.Module):
+    def __init__(self, c_in, c_out, k=1, s=1, g=1, act=True):
+        super().__init__()
+        self.conv = nn.Conv2d(c_in, c_out, k, s, _autopad(k), groups=g, bias=False)
+        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
+        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
+
+    def forward(self, x):
+        return self.act(self.bn(self.conv(x)))
+
+
+class DSConv(nn.Module):
+    """Depthwise separable convolution"""
+
+    def __init__(self, c_in, c_out, k=3, s=1, act=True):
+        super().__init__()
+        self.dw = nn.Conv2d(c_in, c_in, k, s, _autopad(k), groups=c_in, bias=False)
+        self.dw_bn = nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.03)
+        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
+        self.pw_bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
+        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
+
+    def forward(self, x):
+        x = self.act(self.dw_bn(self.dw(x)))
+        x = self.act(self.pw_bn(self.pw(x)))
+        return x
+
+
+class DSBottleneck(nn.Module):
+    def __init__(self, c, k=3, expansion=0.5):
+        super().__init__()
+        hidden = max(8, int(c * expansion))
+        self.cv1 = ConvBNAct(c, hidden, k=1, s=1)
+        self.dw = DSConv(hidden, hidden, k=k, s=1)
+        self.cv2 = ConvBNAct(hidden, c, k=1, s=1)
+
+    def forward(self, x):
+        return x + self.cv2(self.dw(self.cv1(x)))
+
+
+class C3k2Lite(nn.Module):
+    def __init__(self, c, n=1, use_k5=True):
+        super().__init__()
+        blocks = []
+        for i in range(n):
+            k = 5 if (use_k5 and i % 2 == 1) else 3
+            blocks.append(DSBottleneck(c, k=k, expansion=0.5))
+        self.m = nn.Sequential(*blocks) if blocks else nn.Identity()
+
+    def forward(self, x):
+        return self.m(x)
+
+
+class SPPF_Tiny(nn.Module):
+    def __init__(self, c):
+        super().__init__()
+        self.cv1 = ConvBNAct(c, c, k=1, s=1)
+        self.pool = nn.MaxPool2d(5, 1, 2)
+        self.cv2 = ConvBNAct(c, c, k=1, s=1)
+
+    def forward(self, x):
+        return self.cv2(self.pool(self.cv1(x)))
+
+
+class PSALite(nn.Module):
+    def __init__(self, k=7):
+        super().__init__()
+        self.conv = nn.Conv2d(2, 1, k, 1, _autopad(k), bias=True)
+
+    def forward(self, x):
+        avg = torch.mean(x, dim=1, keepdim=True)
+        mx, _ = torch.max(x, dim=1, keepdim=True)
+        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
+        return x * attn
+
+
+class BackboneLite(nn.Module):
+    def __init__(self, in_ch=3, c1=32, c2=64, c3=96):
+        super().__init__()
+        self.stem = ConvBNAct(in_ch, c1, k=3, s=2)
+        self.stage1 = C3k2Lite(c1, n=1)
+        self.down2 = DSConv(c1, c2, k=3, s=2)
+        self.stage2 = C3k2Lite(c2, n=2)
+        self.down3 = DSConv(c2, c3, k=3, s=2)
+        self.stage3 = C3k2Lite(c3, n=3)
+        self.sppf = SPPF_Tiny(c3)
+        self.out_c = c3
+
+    def forward(self, x):
+        x = self.stem(x)
+        x = self.stage1(x)
+        x = self.down2(x)
+        x = self.stage2(x)
+        x = self.down3(x)
+        x = self.stage3(x)
+        return self.sppf(x)
+
+
+class NeckPANLite(nn.Module):
+    def __init__(self, c_in=96, c_mid=32, use_psa=True):
+        super().__init__()
+        self.lateral = ConvBNAct(c_in, c_mid, k=1, s=1)
+        self.merge = C3k2Lite(c_mid, n=1)
+        self.psa = PSALite(k=7) if use_psa else nn.Identity()
+        self.out_c = c_mid
+
+    def forward(self, p3):
+        y = self.lateral(p3)
+        y = self.merge(y)
+        return self.psa(y)
+
+
+class DecoupledHead(nn.Module):
+    def __init__(self, c_in=32, num_classes=1, use_dfl=False, bins=8):
+        super().__init__()
+        self.use_dfl = use_dfl
+        self.bins = bins
+
+        def make_set():
+            stem = ConvBNAct(c_in, c_in, k=3, s=1)
+            reg = nn.Sequential(
+                ConvBNAct(c_in, c_in, k=3, s=1),
+                nn.Conv2d(c_in, (4 if not use_dfl else 4 * bins), 1),
+            )
+            obj = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, 1, 1))
+            cls = nn.Sequential(
+                ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, num_classes, 1)
+            )
+            return stem, reg, obj, cls
+
+        self.stem_m, self.reg_m, self.obj_m, self.cls_m = make_set()
+        self.stem_o, self.reg_o, self.obj_o, self.cls_o = make_set()
+
+    def forward(self, x):
+        xm = self.stem_m(x)
+        reg_m = self.reg_m(xm)
+        obj_m = self.obj_m(xm)
+        cls_m = self.cls_m(xm)
+
+        xo = self.stem_o(x)
+        reg_o = self.reg_o(xo)
+        obj_o = self.obj_o(xo)
+        cls_o = self.cls_o(xo)
+
+        return {"o2m": (reg_m, obj_m, cls_m), "o2o": (reg_o, obj_o, cls_o)}
+
+
+class LP_YOLO_Fuse(nn.Module):
+    def __init__(self, in_ch=3, num_classes=1, use_psa=True, use_dfl=False, bins=8):
+        super().__init__()
+        self.backbone = BackboneLite(in_ch=in_ch)
+        self.neck = NeckPANLite(c_in=self.backbone.out_c, c_mid=32, use_psa=use_psa)
+        self.head = DecoupledHead(
+            c_in=self.neck.out_c, num_classes=num_classes, use_dfl=use_dfl, bins=bins
+        )
+        self.stride = 8
+
+    def forward(self, x):
+        p3 = self.backbone(x)
+        feats = self.neck(p3)
+        return self.head(feats)
+
+
+def dist2bbox_ltbr(ltrb, cx, cy):
+    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
+    x1 = cx - l
+    y1 = cy - t
+    x2 = cx + r
+    y2 = cy + b
+    return torch.stack([x1, y1, x2, y2], dim=-1)
+
+
+def bbox_iou_xyxy(a, b, eps=1e-9):
+    x1 = torch.max(a[..., 0], b[..., 0])
+    y1 = torch.max(a[..., 1], b[..., 1])
+    x2 = torch.min(a[..., 2], b[..., 2])
+    y2 = torch.min(a[..., 3], b[..., 3])
+    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
+    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
+    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
+    return inter / (area_a + area_b - inter + eps)
+
+
+# ---------------------------------------------------------------------------
+# OCR 모델 (AttnOCR) 정의
+# ---------------------------------------------------------------------------
+
+
+class Charset:
+    def __init__(self, charset: str):
+        self.idx2ch = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(charset)
+        self.ch2idx = {ch: i for i, ch in enumerate(self.idx2ch)}
+        self.pad, self.sos, self.eos, self.unk = 0, 1, 2, 3
+
+    def encode(self, text: str, max_len: int) -> torch.Tensor:
+        ids = [self.sos]
+        ids += [self.ch2idx.get(ch, self.unk) for ch in text][: max_len - 2]
+        ids.append(self.eos)
+        if len(ids) < max_len:
+            ids += [self.pad] * (max_len - len(ids))
+        return torch.tensor(ids, dtype=torch.long)
+
+    def decode(self, ids: List[int]) -> str:
+        chars: List[str] = []
+        for idx in ids:
+            ch = self.idx2ch[idx]
+            if ch == "<EOS>":
+                break
+            if ch not in {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}:
+                chars.append(ch)
+        return "".join(chars)
+
+
+class CNNEncoder(nn.Module):
+    def __init__(self, in_ch=1, out_ch=256):
+        super().__init__()
+        self.body = nn.Sequential(
+            nn.Conv2d(in_ch, 64, 3, 1, 1),
+            nn.BatchNorm2d(64),
+            nn.ReLU(True),
+            nn.MaxPool2d(2, 2),
+            nn.Conv2d(64, 128, 3, 1, 1),
+            nn.BatchNorm2d(128),
+            nn.ReLU(True),
+            nn.MaxPool2d(2, 2),
+            nn.Conv2d(128, out_ch, 3, 1, 1),
+            nn.BatchNorm2d(out_ch),
+            nn.ReLU(True),
+        )
+        self.out_ch = out_ch
+
+    def forward(self, x):
+        feat = self.body(x)
+        n, c, h, w = feat.shape
+        feat = F.adaptive_avg_pool2d(feat, (1, w)).squeeze(2)
+        feat = feat.permute(2, 0, 1)
+        return feat
+
+
+class SeqEncoder(nn.Module):
+    def __init__(self, in_dim, hid=256, num_layers=2, bidir=True):
+        super().__init__()
+        self.rnn = nn.LSTM(in_dim, hid, num_layers=num_layers, bidirectional=bidir)
+        self.out_dim = hid * (2 if bidir else 1)
+
+    def forward(self, x):
+        out, _ = self.rnn(x)
+        return out
+
+
+class AdditiveAttention(nn.Module):
+    def __init__(self, enc_dim, dec_dim, attn_dim=256):
+        super().__init__()
+        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
+        self.W_d = nn.Linear(dec_dim, attn_dim, bias=False)
+        self.v = nn.Linear(attn_dim, 1, bias=False)
+
+    def forward(self, enc_out, dec_h):
+        t, n, _ = enc_out.size()
+        e_proj = self.W_e(enc_out)
+        d_proj = self.W_d(dec_h).unsqueeze(0)
+        scores = self.v(torch.tanh(e_proj + d_proj)).squeeze(-1)
+        alpha = torch.softmax(scores, dim=0)
+        ctx = (alpha.unsqueeze(-1) * enc_out).sum(0)
+        return ctx, alpha
+
+
+class AttnDecoder(nn.Module):
+    def __init__(self, vocab_size, enc_dim=256, dec_dim=256, emb_dim=128):
+        super().__init__()
+        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
+        self.rnn = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
+        self.attn = AdditiveAttention(enc_dim, dec_dim)
+        self.fc = nn.Linear(dec_dim + enc_dim, vocab_size)
+
+    def forward(self, enc_out, tgt=None, max_len=16, teacher_forcing=0.5):
+        t, n, _ = enc_out.size()
+        device = enc_out.device
+
+        h = torch.zeros(n, self.rnn.hidden_size, device=device)
+        c = torch.zeros_like(h)
+        y = torch.full((n,), 1, dtype=torch.long, device=device)
+
+        logits = []
+        for step in range(max_len):
+            emb = self.embedding(y)
+            ctx, _ = self.attn(enc_out, h)
+            rnn_in = torch.cat([emb, ctx], dim=-1)
+            h, c = self.rnn(rnn_in, (h, c))
+            out = torch.cat([h, ctx], dim=-1)
+            logit = self.fc(out)
+            logits.append(logit)
+
+            if tgt is not None and torch.rand(1).item() < teacher_forcing:
+                y = tgt[:, step]
+            else:
+                y = logit.argmax(dim=-1)
+
+        logits = torch.stack(logits, dim=1)
+        return logits, None
+
+
+class AttnOCR(nn.Module):
+    def __init__(self, vocab_size, img_ch=1, cnn_dim=256, use_bilstm=True):
+        super().__init__()
+        self.cnn = CNNEncoder(in_ch=img_ch, out_ch=cnn_dim)
+        if use_bilstm:
+            self.seq = SeqEncoder(cnn_dim, hid=256, bidir=True)
+            enc_dim = self.seq.out_dim
+        else:
+            self.seq = nn.Identity()
+            enc_dim = cnn_dim
+        self.dec = AttnDecoder(vocab_size=vocab_size, enc_dim=enc_dim, dec_dim=256, emb_dim=128)
+
+    def forward(self, x, tgt_ids=None, max_len=16, teacher_forcing=0.5):
+        enc = self.cnn(x)
+        enc = self.seq(enc)
+        logits, atts = self.dec(enc, tgt=tgt_ids, max_len=max_len, teacher_forcing=teacher_forcing)
+        return logits, atts
+
+LP_STRIDE = 8  # LP_YOLO_Fuse stride
+DEFAULT_DET_IMG_SIZE = 512
+DEFAULT_DET_CONF = 0.3
+DEFAULT_DET_IOU = 0.5
+DEFAULT_OCR_IMG_SIZE = (32, 128)  # (H, W)
+DEFAULT_OCR_MAX_LEN = 16
+
+PLATE_DET_WEIGHT_ENV = "PLATE_DET_WEIGHTS"
+OCR_WEIGHT_ENV = "PLATE_OCR_WEIGHTS"
+DEFAULT_DET_WEIGHTS = "plate_detector.pt"
+DEFAULT_OCR_WEIGHTS = "ocr_model.pt"
+WEIGHT_CONFIG_FILE = BASE_DIR / "weights_config.json"
+
+
+@dataclass
+class OCRBundle:
+    model: torch.nn.Module
+    charset: Charset
+    img_size: Tuple[int, int]
+    max_length: int
+
+
+def _load_weight_config() -> dict:
+    """weights_config.json을 읽어 가중치 경로를 가져온다."""
+    config = {
+        "plate_detector": DEFAULT_DET_WEIGHTS,
+        "ocr": DEFAULT_OCR_WEIGHTS,
+    }
+    if WEIGHT_CONFIG_FILE.exists():
+        try:
+            with WEIGHT_CONFIG_FILE.open("r", encoding="utf-8") as f:
+                data = json.load(f)
+            if isinstance(data, dict):
+                for key in ("plate_detector", "ocr"):
+                    if key in data and isinstance(data[key], str):
+                        config[key] = data[key]
+        except json.JSONDecodeError as exc:
+            raise ValueError(
+                f"Invalid JSON format in {WEIGHT_CONFIG_FILE}: {exc}"
+            ) from exc
+    return config
+
+
+def _resolve_weight_path(env_key: str, config_key: str, config: dict) -> Path:
+    env_override = os.environ.get(env_key)
+    if env_override:
+        candidate = Path(env_override)
+    else:
+        candidate = Path(config.get(config_key, ""))
+    if not candidate.is_absolute():
+        candidate = BASE_DIR / candidate
+    return candidate
+
+
+def _load_plate_detector(weight_path: Path, device: torch.device) -> torch.nn.Module:
+    if not weight_path.exists():
+        raise FileNotFoundError(
+            f"Plate detector weights not found at '{weight_path}'. "
+            f"Place the trained checkpoint there or set {PLATE_DET_WEIGHT_ENV}"
+        )
+
+    model = LP_YOLO_Fuse().to(device)
+    ckpt = torch.load(weight_path, map_location=device)
+    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
+        state_dict = ckpt["state_dict"]
+    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
+        state_dict = ckpt
+    else:
+        state_dict = ckpt
+
+    model.load_state_dict(state_dict, strict=False)
+    model.eval()
+    return model
+
+
+def _load_ocr_bundle(weight_path: Path, device: torch.device) -> OCRBundle:
+    if not weight_path.exists():
+        raise FileNotFoundError(
+            f"OCR weights not found at '{weight_path}'. "
+            f"Place the trained checkpoint there or set {OCR_WEIGHT_ENV}"
+        )
+
+    ckpt = torch.load(weight_path, map_location=device)
+    if not isinstance(ckpt, dict) or "model" not in ckpt:
+        raise ValueError("OCR checkpoint must contain a 'model' state dictionary")
+
+    charset_tokens = ckpt.get("charset")
+    if charset_tokens:
+        if isinstance(charset_tokens, (list, tuple)):
+            charset_chars = "".join(
+                ch for ch in charset_tokens if ch not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
+            )
+        else:
+            charset_chars = str(charset_tokens)
+    else:
+        # 기본 숫자/영문/한글 일부 – 필요 시 가중치에 포함된 charset으로 교체
+        charset_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하"
+
+    charset = Charset(charset_chars)
+    ocr_model = AttnOCR(
+        vocab_size=len(charset.idx2ch), img_ch=1, cnn_dim=256, use_bilstm=True
+    ).to(device)
+    ocr_model.load_state_dict(ckpt["model"], strict=True)
+    ocr_model.eval()
+
+    img_size = tuple(ckpt.get("img_hw", DEFAULT_OCR_IMG_SIZE))
+    max_len = int(ckpt.get("max_len", DEFAULT_OCR_MAX_LEN))
+
+    return OCRBundle(model=ocr_model, charset=charset, img_size=img_size, max_length=max_len)
+
+
 # ==============================
 # 모델 로드
 # ==============================
 def load_models():
     device = torch.device("cpu")
-    # 차량 감지 모델
-    car_m = torch.hub.load('./yolov5', 'yolov5s', source='local').to(device)
-    # 번호판 감지 모델 (커스텀)
-    lp_m = torch.hub.load('./yolov5', 'custom', path='lp_det.pt', source='local').to(device)
-    # OCR 모델
-    ocr_model = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False)
-    # 차량 클래스만 필터링 (2: car, 3: motorcycle 등)
-    car_m.classes = [2, 3, 5, 7]
-    return car_m, lp_m, ocr_model
+    config = _load_weight_config()
+    det_weights = _resolve_weight_path(PLATE_DET_WEIGHT_ENV, "plate_detector", config)
+    ocr_weights = _resolve_weight_path(OCR_WEIGHT_ENV, "ocr", config)
+
+    plate_detector = _load_plate_detector(det_weights, device)
+    ocr_bundle = _load_ocr_bundle(ocr_weights, device)
+    return plate_detector, ocr_bundle
+
+
+def _build_grid(height: int, width: int, stride: int, device: torch.device):
+    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
+    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
+    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
+    return cx * stride, cy * stride
+
+
+def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
+    if boxes.numel() == 0:
+        return torch.empty(0, dtype=torch.long, device=boxes.device)
+
+    order = scores.argsort(descending=True)
+    keep: List[int] = []
+    while order.numel() > 0:
+        i = order[0].item()
+        keep.append(i)
+        if order.numel() == 1:
+            break
+        rest = order[1:]
+        ious = bbox_iou_xyxy(boxes[rest], boxes[i].unsqueeze(0))
+        if ious.ndim > 1:
+            ious = ious.squeeze(-1)
+        order = rest[ious <= iou_threshold]
+    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
+
+
+def _run_plate_detection(
+    image_rgb: np.ndarray,
+    model: torch.nn.Module,
+    img_size: int = DEFAULT_DET_IMG_SIZE,
+    conf_thr: float = DEFAULT_DET_CONF,
+    iou_thr: float = DEFAULT_DET_IOU,
+) -> Tuple[np.ndarray, np.ndarray]:
+    device = next(model.parameters()).device
+    orig_h, orig_w = image_rgb.shape[:2]
+    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
+    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
+    tensor = tensor.to(device)
+
+    with torch.no_grad():
+        outputs = model(tensor)
+        reg_o, obj_o, cls_o = outputs["o2o"]
+
+    B, _, H, W = obj_o.shape
+    cx, cy = _build_grid(H, W, getattr(model, "stride", LP_STRIDE), tensor.device)
+
+    if reg_o.shape[1] == 4:
+        reg = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(min=0)
+    else:
+        bins = reg_o.shape[1] // 4
+        reg_logits = reg_o.permute(0, 2, 3, 1).reshape(B, -1, 4, bins)
+        prob = torch.softmax(reg_logits, dim=-1)
+        idx = torch.arange(bins, device=reg_logits.device, dtype=reg_logits.dtype)
+        reg = (prob * idx).sum(-1)
+
+    boxes = dist2bbox_ltbr(reg, cx.view(-1), cy.view(-1)).view(B, -1, 4)
+    scores = torch.sigmoid(obj_o).view(B, -1) * torch.sigmoid(cls_o).view(B, -1)
+
+    boxes = boxes[0]
+    scores = scores[0]
+    valid = scores > conf_thr
+    boxes = boxes[valid]
+    scores = scores[valid]
+
+    keep = _nms_indices(boxes, scores, iou_thr)
+    if keep.numel() > 0:
+        boxes = boxes[keep]
+        scores = scores[keep]
+        order = scores.argsort(descending=True)
+        boxes = boxes[order]
+        scores = scores[order]
+    else:
+        boxes = boxes.new_zeros((0, 4))
+        scores = scores.new_zeros((0,))
+
+    if boxes.numel() == 0:
+        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
+
+    scale_x = orig_w / float(img_size)
+    scale_y = orig_h / float(img_size)
+    boxes = boxes.clone()
+    boxes[:, [0, 2]] *= scale_x
+    boxes[:, [1, 3]] *= scale_y
+    boxes = boxes.cpu().numpy()
+    scores = scores.cpu().numpy()
+    return boxes, scores
+
+
+def _clean_text(text: str) -> str:
+    return text.replace(',', '').replace('-', '').strip()
+
+
+def _run_ocr_on_crop(crop_rgb: np.ndarray, bundle: OCRBundle) -> str:
+    model = bundle.model
+    device = next(model.parameters()).device
+    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
+    h, w = bundle.img_size
+    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
+    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
+
+    with torch.no_grad():
+        logits, _ = model(tensor, tgt_ids=None, max_len=bundle.max_length, teacher_forcing=0.0)
+    ids = logits.argmax(-1)[0].tolist()
+    text = bundle.charset.decode(ids)
+    return _clean_text(text)
+
+
+def _save_plate_crop(crop_rgb: np.ndarray) -> str:
+    crops_dir = BASE_DIR / "plate_crops"
+    crops_dir.mkdir(exist_ok=True)
+    file_path = crops_dir / f"plate_{uuid.uuid4().hex}.png"
+    cv2.imwrite(str(file_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
+    return str(file_path)
+
 
 # ==============================
 # OCR/번호판 추론 함수
 # ==============================
-def deskew_plate(plate_img):
-    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
-    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
-    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
-    if lines is None:
-        return plate_img
-    angles = [(theta*180/np.pi - 90) for rho, theta in (l[0] for l in lines) if -45 < theta*180/np.pi-90 < 45]
-    if not angles:
-        return plate_img
-    median_angle = np.median(angles)
-    h, w = plate_img.shape[:2]
-    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
-    rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
-    return rotated
-
-def group_by_chars(ocr_result, y_thresh=10):
-    lines = []
-    for box, (text, _) in ocr_result:
-        y_center = (box[0][1] + box[2][1]) / 2
-        matched = False
-        for line in lines:
-            if abs(line[0][0] - y_center) < y_thresh:
-                line.extend([(y_center, box, ch) for ch in text])
-                matched = True
-                break
-        if not matched:
-            lines.append([(y_center, box, ch) for ch in text])
-    lines.sort(key=lambda x: x[0][0])
-    sorted_texts = []
-    for line in lines:
-        line.sort(key=lambda x: x[1][0][0])
-        sorted_texts.append(''.join([t[2] for t in line]))
-    return sorted_texts
-
-def detect_car_plate(img_path, car_m, lp_m, ocr_model):
-    """
-    번호판 인식 + 번호판 크롭 이미지 리턴
-    """
+def detect_car_plate(img_path: str, plate_detector: torch.nn.Module, ocr_bundle: OCRBundle):
+    """번호판 인식 + 번호판 크롭 이미지 리턴"""
     try:
         im_pil = Image.open(img_path).convert("RGB")
     except (UnidentifiedImageError, OSError):
         return ["인식 불가 (이미지 파일 아님)"], []
 
     img = np.array(im_pil)
-    result_text = []
-    plate_imgs = []
-
-    locs = car_m(im_pil).xyxy[0]
-    if len(locs) > 0:
-        for item in locs:
-            x1, y1, x2, y2 = [int(t.cpu().detach().numpy()) for t in item[:4]]
-            car_crop = img[y1:y2, x1:x2, :].copy()
-            lp_results = lp_m(Image.fromarray(car_crop))
-            for lp in lp_results.xyxy[0]:
-                lx1, ly1, lx2, ly2 = [int(t.cpu().detach().numpy()) for t in lp[:4]]
-                plate_crop = car_crop[ly1:ly2, lx1:lx2].copy()
-                # deskew 후 gray 변환
-                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
-                plate_imgs.append(gray_plate)
-                ocr_results = ocr_model.ocr(cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR), cls=True)
-                if ocr_results and ocr_results[0]:
-                    text = ''.join([t[1][0] for t in ocr_results[0]]).replace(',', '').replace('-', '')
-                    result_text.append(text)
-
-    if not result_text:
-        # 전체 이미지 OCR fallback
-        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
-        ocr_full = ocr_model.ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cls=True)
-        if ocr_full and ocr_full[0]:
-            result_text = [''.join([t[1][0] for t in ocr_full[0]]).replace(',', '').replace('-', '')]
-        else:
-            result_text = ["인식 실패"]
+    texts: List[str] = []
+    plate_imgs: List[str] = []
+
+    boxes, _ = _run_plate_detection(img, plate_detector)
+    for box in boxes:
+        x1, y1, x2, y2 = [int(round(v)) for v in box]
+        x1 = max(0, min(img.shape[1] - 1, x1))
+        y1 = max(0, min(img.shape[0] - 1, y1))
+        x2 = max(0, min(img.shape[1], x2))
+        y2 = max(0, min(img.shape[0], y2))
+        if x2 <= x1 or y2 <= y1:
+            continue
+        crop = img[y1:y2, x1:x2].copy()
+        crop_path = _save_plate_crop(crop)
+        plate_imgs.append(crop_path)
+        text = _run_ocr_on_crop(crop, ocr_bundle)
+        if text:
+            texts.append(text)
+
+    if not texts:
+        fallback = _run_ocr_on_crop(img, ocr_bundle)
+        texts = [fallback] if fallback else ["인식 실패"]
 
-    return result_text, plate_imgs
+    return texts, plate_imgs
 
EOF
)
