import contextlib
import importlib.util
import io
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
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
OBJ_SRC_DIR = BASE_DIR / "Obj Detection" / "src"
OCR_SRC_PATH = BASE_DIR / "OCR" / "ocr_model.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_lp_model_mod = None
_lp_utils_mod = None
_ocr_mod = None


def _get_detection_modules():
    global _lp_model_mod, _lp_utils_mod
    if _lp_model_mod is None or _lp_utils_mod is None:
        if not OBJ_SRC_DIR.exists():
            raise FileNotFoundError("Obj Detection/src 폴더가 없습니다. 학습한 번호판 검출 코드가 필요합니다.")
        _lp_model_mod = _load_module("lp_model", OBJ_SRC_DIR / "model.py")
        _lp_utils_mod = _load_module("lp_utils", OBJ_SRC_DIR / "utils_common.py")
    return _lp_model_mod, _lp_utils_mod


def _get_ocr_module():
    global _ocr_mod
    if _ocr_mod is None:
        if not OCR_SRC_PATH.exists():
            raise FileNotFoundError("OCR/ocr_model.py를 찾을 수 없습니다. 학습한 OCR 코드가 필요합니다.")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ocr_mod = _load_module("ocr_model", OCR_SRC_PATH)
    return _ocr_mod


@dataclass
class OCRBundle:
    model: torch.nn.Module
    charset: object
    max_length: int
    img_size: Tuple[int, int]


DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하서울부산대구인천광주대전울산경기강원충북충남전북전남경북경남제주세종"
DEFAULT_DET_IMG_SIZE = 512
DEFAULT_DET_CONF = 0.3
DEFAULT_DET_IOU = 0.4
LP_STRIDE = 8


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
    model_mod, utils_mod = _get_detection_modules()
    model_cls = getattr(model_mod, "LP_YOLO_Fuse")
    model = model_cls().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if not hasattr(model, "stride"):
        model.stride = LP_STRIDE
    model.dist2bbox = getattr(utils_mod, "dist2bbox_ltbr")
    return model


def _load_ocr_bundle(weight_path: Path, device: torch.device) -> OCRBundle:
    ocr_mod = _get_ocr_module()
    Charset = getattr(ocr_mod, "Charset")
    AttnOCR = getattr(ocr_mod, "AttnOCR")

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

    dist2bbox = getattr(model, "dist2bbox", None)
    if dist2bbox is None:
        _, utils_mod = _get_detection_modules()
        dist2bbox = getattr(utils_mod, "dist2bbox_ltbr")
    boxes = dist2bbox(reg, cx.view(-1), cy.view(-1)).view(B, -1, 4)
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
