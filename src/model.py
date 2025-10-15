import torch, torch.nn as nn
from .utils_common import ConvBNAct, DSConv, C3k2Lite, SPPF_Tiny, PSALite

# Backbone: Conv(Stem) → [C3k2Lite(DS) → DSConv s=2] ×2 → C3k2Lite → SPPF_Tiny  (P3-only)
class BackboneLite(nn.Module):
    def __init__(self, in_ch=3, c1=32, c2=64, c3=96):
        super().__init__()
        self.stem   = ConvBNAct(in_ch, c1, k=3, s=2)   # 1/2
        self.stage1 = C3k2Lite(c1, n=1)
        self.down2  = DSConv(c1, c2, k=3, s=2)         # 1/4
        self.stage2 = C3k2Lite(c2, n=2)
        self.down3  = DSConv(c2, c3, k=3, s=2)         # 1/8
        self.stage3 = C3k2Lite(c3, n=3)
        self.sppf   = SPPF_Tiny(c3)
        self.out_c  = c3
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x)
        x = self.down2(x); x = self.stage2(x)
        x = self.down3(x); x = self.stage3(x)
        return self.sppf(x)   # P3

# Neck: Lateral 1x1(96→32) → C3k2Lite(DS, n=1) → PSA-Lite ×1
class NeckPANLite(nn.Module):
    def __init__(self, c_in=96, c_mid=32, use_psa=True):
        super().__init__()
        self.lateral = ConvBNAct(c_in, c_mid, k=1, s=1)     # STD
        self.merge   = C3k2Lite(c_mid, n=1)                 # DS
        self.psa     = PSALite(k=7) if use_psa else nn.Identity()
        self.out_c   = c_mid
    def forward(self, p3):
        y = self.lateral(p3)
        y = self.merge(y)
        return self.psa(y)

# Head: Decoupled + Dual (o2m & o2o)  — DFL은 토글로 추가 가능
class DecoupledHead(nn.Module):
    def __init__(self, c_in=32, num_classes=1, use_dfl=False, bins=8):
        super().__init__()
        self.use_dfl, self.bins = use_dfl, bins
        def make_set():
            stem = ConvBNAct(c_in, c_in, k=3, s=1)
            reg  = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1),
                                 nn.Conv2d(c_in, (4 if not use_dfl else 4*bins), 1))
            obj  = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, 1, 1))
            cls  = nn.Sequential(ConvBNAct(c_in, c_in, k=3, s=1), nn.Conv2d(c_in, num_classes, 1))
            return stem, reg, obj, cls
        self.stem_m, self.reg_m, self.obj_m, self.cls_m = make_set()
        self.stem_o, self.reg_o, self.obj_o, self.cls_o = make_set()

    def forward(self, x):
        xm = self.stem_m(x); reg_m = self.reg_m(xm); obj_m = self.obj_m(xm); cls_m = self.cls_m(xm)
        xo = self.stem_o(x); reg_o = self.reg_o(xo); obj_o = self.obj_o(xo); cls_o = self.cls_o(xo)
        return {"o2m": (reg_m, obj_m, cls_m), "o2o": (reg_o, obj_o, cls_o)}

class LP_YOLO_Fuse(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, use_psa=True, use_dfl=False, bins=8):
        super().__init__()
        self.backbone = BackboneLite(in_ch=in_ch)
        self.neck     = NeckPANLite(c_in=self.backbone.out_c, c_mid=32, use_psa=use_psa)
        self.head     = DecoupledHead(c_in=self.neck.out_c, num_classes=num_classes, use_dfl=use_dfl, bins=bins)
        self.stride   = 8  # P3-only
    def forward(self, x):
        p3 = self.backbone(x)
        f  = self.neck(p3)
        return self.head(f)
