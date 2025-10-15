import torch, argparse
from .model import LP_YOLO_Fuse
from .utils_common import dist2bbox_ltbr

def build_grid(H, W, stride, device):
    ys = torch.arange(H, device=device) + 0.5
    xs = torch.arange(W, device=device) + 0.5
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')
    return (cx*stride, cy*stride)

@torch.no_grad()
def infer(img, model, conf_thr=0.2):
    out = model(img)
    (reg_m, obj_m, cls_m), (reg_o, obj_o, cls_o) = out["o2m"], out["o2o"]
    B,_,H,W = obj_o.shape; s = model.stride
    cx, cy = build_grid(H,W,s, img.device)
    reg = reg_o.permute(0,2,3,1).reshape(B,-1,4).clamp(min=0)
    boxes = dist2bbox_ltbr(reg, cx.view(-1), cy.view(-1)).view(B,-1,4)
    score = torch.sigmoid(obj_o).view(B,-1,1) * torch.sigmoid(cls_o).view(B,-1,1)
    keep = score[...,0] > conf_thr
    out_boxes = [boxes[b][keep[b]] for b in range(B)]
    out_scores= [score[b][keep[b]][:,0] for b in range(B)]
    return out_boxes, out_scores

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--imgsz", type=int, default=512); args = ap.parse_args()
    model = LP_YOLO_Fuse().eval()
    img = torch.zeros(1,3,args.imgsz,args.imgsz)
    boxes, scores = infer(img, model, conf_thr=0.2)
    print("detections:", [b.shape for b in boxes], [s.shape for s in scores])

if __name__ == "__main__":
    main()
