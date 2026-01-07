import torch
import torch.nn as nn

class Fire(nn.Module):
    """
    Fire Module from SqueezeNet.
    Structure: Squeeze (1x1) -> Expand (1x1 + 3x3)
    """
    def __init__(self, c1, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        # Squeeze layer
        self.squeeze = nn.Conv2d(c1, squeeze_planes, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
            self.expand1x1_act(self.expand1x1(x)),
            self.expand3x3_act(self.expand3x3(x))
        ], 1)