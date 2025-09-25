from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResBlock(nn.Module):
    def __init__(self, nf: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.res_scale = res_scale
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        return x + self.res_scale * (self.c2(F.relu(self.c1(x), inplace=True)))

class EDSRLite(nn.Module):
    """
    Restoration 1 x (LR & HR cùng kích thước).
    Input:  [B, 3, H, W]  in [0, 1]
    Output: [B, 3, H, W]  in [0, 1]
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3, nf: int = 64, nb: int = 12, res_scale: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_ResBlock(nf, res_scale) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.head(x)      # [B, nf, H, W]
        y = self.body(s)      # [B, nf, H, W]
        y = self.tail(y + s)  # [B, 3, H, W]
        return y.clamp(0, 1)
