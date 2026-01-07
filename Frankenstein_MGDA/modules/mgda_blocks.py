import torch
import torch.nn as nn
from torch.autograd import Function

# --- PHASE 2: Gradient Reversal Layer (GRL) ---
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainAdversarialHead(nn.Module):
    """
    Attached to the Main Head. Tries to classify Domain (Source vs Target).
    Includes the GRL to flip gradients during backprop.
    """
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.grl = GradientReversalFn.apply
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 2) # Output: 2 classes (Source=0, Target=1)
        )

    def forward(self, x, alpha=1.0):
        x = self.grl(x, alpha)
        return self.block(x)


# --- PHASE 4: Super-Resolution Decoder ---
class SuperResolutionDecoder(nn.Module):
    """
    Attached to the Backbone (Layer 9). Tries to reconstruct the original image.
    Forces the backbone to learn high-res structural details.
    """
    def __init__(self, in_channels, out_channels=3, scale_factor=4):
        super().__init__()
        # Upsample 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True)
        )
        # Upsample 2
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True)
        )
        # Final Reconstruction
        self.final = nn.Conv2d(in_channels // 4, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)



# --- PHASE 3: Attention Alignment (Entropy Minimization) ---
class AttentionConsistencyLoss(nn.Module):
    """
    Calculates the Entropy of the Feature Maps from the C2PSA block.
    Goal: Force the model to be 'confident' (Low Entropy) about where it looks,
    regardless of the domain (Source or Target).
    """
    def __init__(self):
        super().__init__()

    def forward(self, feature_map):
        """
        Args:
            feature_map: Output from C2PSA block [Batch, Channel, Height, Width]
        Returns:
            entropy_loss: Scalar value
        """
        # 1. Spatial Softmax (normalize across HxW to get a probability map)
        b, c, h, w = feature_map.size()
        x = feature_map.view(b, c, -1) # Flatten spatial dims
        x = torch.softmax(x, dim=2)    # Softmax over spatial pixels

        # 2. Calculate Entropy: -sum(p * log(p))
        # We add 1e-6 to avoid log(0) error
        entropy = -torch.sum(x * torch.log(x + 1e-6), dim=2)
        
        # 3. Return mean entropy over the batch
        return entropy.mean()
    