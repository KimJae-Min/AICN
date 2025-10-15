# -*- coding: utf-8 -*-
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

from src.model import LP_YOLO_Fuse
from src.utils_common import dist2bbox_ltbr

def load_img(path, imgsz):
    im = Image.open(path).convert("RGB")
    orig = im.copy()
    im = im.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).contiguous()
    return img, orig

def build_grid_xy(H,W,stride,device):
    ys = torch.arange(H, device=device) + 0.5
    xs = torch.arange(W, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    return (cx*stride).view(-1), (cy*stride).view(-1)

@torch.no_grad()
def main():
    base = Path(r"C:/Users/win/Desktop/lp_yolo_fuse")
    data = base/"data"
    weights = base/"lp_yolo_fuse_lp.pt"
    imgsz = 512; conf=0.25; iou=0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LP_YOLO_Fuse().to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device))

    vimg = data/"images/val"
    outd = data/"viz_preds"; outd.mkdir(parents=True, exist_ok=True)
    paths = [p for p in vimg.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]]
    random.seed(0)
    for ip in random.sample(paths, k=min(20, len(paths))):
        img, orig = load_img(ip, imgsz)
        out = model(img.to(device))
        (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
        B,_,H,W = obj_o.shape; s=model.stride
        cx, cy = build_grid_xy(H,W,s,device)
        reg = reg_o.permute(0,2,3,1).reshape(1,-1,4).clamp(min=0)[0]
        boxes = dist2bbox_ltbr(reg, cx, cy)
        scores = (torch.sigmoid(obj_o[0,0].flatten()) * torch.sigmoid(cls_o[0,0].flatten()))
        m = scores > conf
        boxes = boxes[m]; scores = scores[m]
        if boxes.numel():
            keep = nms(boxes, scores, iou)
            boxes = boxes[keep].cpu().numpy()
        draw = ImageDraw.Draw(orig)
        for (x1,y1,x2,y2) in boxes:
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        orig.save(outd/f"{ip.stem}.jpg")
    print("saved to", outd)

if __name__ == "__main__":
    main()
