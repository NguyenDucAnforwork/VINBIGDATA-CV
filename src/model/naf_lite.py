from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utils ----------
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW (apply LN over C at each spatial location)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] -> LN(C) -> [B, C, H, W]
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------- NAF Block ----------
class _NAFBlock(nn.Module):
    """
    NAFBlock (simple, activation-free):
      - LN
      - 1x1 Conv (expand to 2*C)
      - 3x3 Depthwise Conv
      - SimpleGate: split along C and multiply
      - SCA: global avg pool -> 1x1 conv -> scale
      - 1x1 Conv (project back to C)
      - Residual with scale
    """
    def __init__(self, nf: int, dw_kernel: int = 3, res_scale: float = 1.0):
        super().__init__()
        self.norm = LayerNorm2d(nf)

        hidden = nf * 2
        self.conv1 = nn.Conv2d(nf, hidden, 1, 1, 0)
        self.dwconv = nn.Conv2d(hidden, hidden, dw_kernel, 1, dw_kernel // 2, groups=hidden)

        # Simplified Channel Attention (SCA)
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(hidden // 2, hidden // 2, 1, 1, 0)  # after SimpleGate halves channels

        self.conv2 = nn.Conv2d(hidden // 2, nf, 1, 1, 0)

        # residual scaling (trainable optional)
        self.res_scale = nn.Parameter(torch.ones(1) * res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        s = self.norm(x)
        u = self.conv1(s)          # [B, 2C, H, W]
        u = self.dwconv(u)         # [B, 2C, H, W]

        # SimpleGate: split and multiply
        u1, u2 = torch.chunk(u, 2, dim=1)  # each [B, C, H, W]
        g = u1 * u2                        # [B, C, H, W]

        # SCA
        w = self.sca_conv(self.sca_pool(g))  # [B, C, 1, 1]
        g = g * w

        y = self.conv2(g)                 # [B, C, H, W]
        return x + self.res_scale * y


# ---------- Model ----------
class NAFLite(nn.Module):
    """
    Restoration 1x (LR & HR cùng kích thước).
    Input:  [B, 3, H, W] in [0, 1]
    Output: [B, 3, H, W] in [0, 1]
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3, nf: int = 64, nb: int = 12, res_scale: float = 1.0):
        super().__init__()
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_NAFBlock(nf, dw_kernel=3, res_scale=res_scale) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.head(x)          # [B, nf, H, W]
        y = self.body(s)          # [B, nf, H, W]
        y = self.tail(y + s)      # long skip
        return y.clamp(0, 1)
