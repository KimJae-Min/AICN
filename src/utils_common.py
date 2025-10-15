import math
import torch
import torch.nn as nn

def autopad(k, p=None):
    if p is None: p = k // 2
    return p

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DSConv(nn.Module):
    """Depthwise-Separable: DW kxk + 1x1(PW)"""
    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, autopad(k), groups=c_in, bias=False)
        self.dw_bn = nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.03)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class DSBottleneck(nn.Module):
    """1x1 -> DW(3/5) -> 1x1 (+ residual)"""
    def __init__(self, c, k=3, expansion=0.5):
        super().__init__()
        ch = max(8, int(c * expansion))
        self.cv1 = ConvBNAct(c, ch, k=1, s=1)
        self.dw  = DSConv(ch, ch, k=k, s=1)
        self.cv2 = ConvBNAct(ch, c, k=1, s=1)
    def forward(self, x):
        return x + self.cv2(self.dw(self.cv1(x)))

class C3k2Lite(nn.Module):
    """DS bottleneck를 얕게 스택 (k=3/5 교차)"""
    def __init__(self, c, n=1, use_k5=True):
        super().__init__()
        layers = []
        for i in range(n):
            k = 5 if (use_k5 and i % 2 == 1) else 3
            layers.append(DSBottleneck(c, k=k, expansion=0.5))
        self.m = nn.Sequential(*layers) if layers else nn.Identity()
    def forward(self, x): return self.m(x)

class SPPF_Tiny(nn.Module):
    """1x1 -> MaxPool(5) -> 1x1"""
    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBNAct(c, c, k=1, s=1)
        self.pool= nn.MaxPool2d(5,1,2)
        self.cv2 = ConvBNAct(c, c, k=1, s=1)
    def forward(self, x): return self.cv2(self.pool(self.cv1(x)))

class PSALite(nn.Module):
    """Avg/Max channel maps -> concat(2) -> 7x7 conv -> sigmoid -> spatial gate"""
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, autopad(k), bias=True)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a

def dist2bbox_ltbr(ltrb, cx, cy):
    """ltrb -> xyxy (broadcast to grid)"""
    l, t, r, b = ltrb[...,0], ltrb[...,1], ltrb[...,2], ltrb[...,3]
    x1 = cx - l; y1 = cy - t; x2 = cx + r; y2 = cy + b
    return torch.stack([x1,y1,x2,y2], dim=-1)
