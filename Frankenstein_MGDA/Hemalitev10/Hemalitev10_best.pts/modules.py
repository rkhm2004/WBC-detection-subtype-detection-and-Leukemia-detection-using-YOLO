import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    def __init__(self, c1, c2=None, reduction=32):
        """Coordinate Attention block.

        Robust to unexpected `reduction` values (avoids division by zero)
        and accepts either one or two channel arguments as some parsers pass
        only `c1`.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        # ensure integer channel counts
        self.c1 = int(c1)
        self.c2 = int(c2)

        # sanitize reduction to avoid ZeroDivisionError
        try:
            reduction_val = int(reduction)
        except Exception:
            reduction_val = 32
        if reduction_val <= 0:
            reduction_val = 1

        mip = max(8, max(1, self.c1 // reduction_val))

        self.conv1 = nn.Conv2d(self.c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, self.c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, self.c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # pooled along width and height (robust without invalid pool sizes)
        x_h = x.mean(dim=3, keepdim=True)               # (n, c, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (n, c, w, 1)
        y = torch.cat([x_h, x_w], dim=2)                # (n, c, h+w, 1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        attn = self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()
        return identity * attn


class ESPP(nn.Module):
    def __init__(self, c1, c2=None):
        """Enhanced spatial pyramid pooling. Accepts optional c2 (defaults to c1)."""
        super().__init__()
        if c2 is None:
            c2 = c1
        c1 = int(c1)
        c2 = int(c2)
        hidden = max(1, c1 // 4)
        self.cv1 = nn.Conv2d(c1, hidden, 1, 1, 0)
        self.cv2 = nn.Conv2d(hidden, hidden, 3, 1, 1)
        self.cv3 = nn.Conv2d(hidden, hidden, 3, 1, 3, dilation=3)
        self.cv4 = nn.Conv2d(hidden, hidden, 3, 1, 5, dilation=5)
        self.cv5 = nn.Conv2d(hidden * 4, c2, 1, 1, 0)

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv5(torch.cat([x1, self.cv2(x1), self.cv3(x1), self.cv4(x1)], dim=1))