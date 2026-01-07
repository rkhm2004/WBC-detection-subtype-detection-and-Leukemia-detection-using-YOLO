import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
from modules.mgda_blocks import DomainAdversarialHead, SuperResolutionDecoder, AttentionConsistencyLoss
import os

# --- CONFIGURATION ---
# Dynamic path setup to avoid FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CFG = os.path.join(BASE_DIR, 'models', 'yolo_frankenstein_mgda.yaml')
DATA_CFG = os.path.join(BASE_DIR, 'datasets', 'bccd_data.yaml') 
PROJECT_NAME = 'Frankenstein_mgda_on_Yolonew'
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16

# --- PICKLE-SAFE HOOK CLASS ---
# This class replaces the 'local function' so torch.save can handle it.
class ActivationHook:
    def __init__(self):
        self.output = None
        
    def __call__(self, module, input, output):
        self.output = output

# --- PHASE 1: FOURIER MIXING ---
def fourier_mix_batch(batch_imgs, alpha=0.1):
    """Phase 1: Spectral Style Swapping"""
    if torch.rand(1) > 0.5: return batch_imgs # 50% chance to apply
    
    # FFT
    fft = torch.fft.fft2(batch_imgs, dim=(-2, -1))
    amp, pha = torch.abs(fft), torch.angle(fft)
    amp_shift = torch.roll(amp, shifts=1, dims=0) # Shuffle batch style
    
    b, c, h, w = batch_imgs.shape
    b_h, b_w = int(h * alpha), int(w * alpha)
    cy, cx = h // 2, w // 2
    
    # Swap Low Freq (Style)
    amp[:, :, cy-b_h:cy+b_h, cx-b_w:cx+b_w] = amp_shift[:, :, cy-b_h:cy+b_h, cx-b_w:cx+b_w]
    
    # Reconstruct
    fft_new = amp * torch.exp(1j * pha)
    return torch.fft.ifft2(fft_new, dim=(-2, -1)).real

# --- CUSTOM TRAINER ---
class MGDATrainer(DetectionTrainer):
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Initialize and Attach Hooks"""
        model = super().get_model(cfg, weights, verbose)
        
        # 1. Initialize Auxiliary Modules
        # Attach them to the model so they are saved with the checkpoint
        model.sr_decoder = SuperResolutionDecoder(in_channels=256).to(self.device)
        model.grl_head = DomainAdversarialHead(in_channels=512).to(self.device)
        model.attn_loss_fn = AttentionConsistencyLoss()
        
        # 2. Register Hooks (Using the Pickle-Safe Class)
        # We attach the hook objects to the model as attributes so we can access them in train_step
        
        # HOOK 1: Phase 4 (SR Decoder) - Layer 8
        model.hook_l8 = ActivationHook()
        model.model[8].register_forward_hook(model.hook_l8)

        # HOOK 2: Phase 3 (Attention) - Layer 12 (P4 Attention Output)
        model.hook_l12 = ActivationHook()
        model.model[12].register_forward_hook(model.hook_l12)
        
        # HOOK 3: Phase 2 (GRL) - Layer 27 (Head Output)
        model.hook_head = ActivationHook()
        model.model[27].register_forward_hook(model.hook_head)
        
        return model

    def preprocess_batch(self, batch):
        """Phase 1: Fourier Mixing"""
        batch = super().preprocess_batch(batch)
        if self.epoch < (self.epochs * 0.8): 
            batch['img'] = fourier_mix_batch(batch['img'])
        return batch

    def train_step(self, batch):
        """
        Overridden Training Step:
        Standard Forward -> Catch Features -> Calc Aux Loss -> Combined Backward
        """
        self.optimizer.zero_grad()
        
        # 1. Prepare Data
        batch = self.preprocess_batch(batch)
        images = batch['img']
        
        # 2. Standard Forward Pass (Hooks will capture features automatically)
        preds = self.model(images)
        
        # 3. Calculate Standard YOLO Loss
        loss, loss_items = self.criterion(preds, batch)
        
        # 4. Calculate MGDA Auxiliary Losses
        aux_loss = 0
        
        # --- Phase 4: SR Decoder Loss ---
        # Access the hook object stored on the model
        feat_l8 = self.model.hook_l8.output
        if feat_l8 is not None:
            reconstructed = self.model.sr_decoder(feat_l8)
            target_resized = F.interpolate(images, size=reconstructed.shape[2:], mode='bilinear')
            sr_loss = F.mse_loss(reconstructed, target_resized)
            aux_loss += (0.1 * sr_loss) 
            
        # --- Phase 3: Attention Consistency Loss ---
        feat_l12 = self.model.hook_l12.output
        if feat_l12 is not None:
            attn_loss = self.model.attn_loss_fn(feat_l12)
            aux_loss += (0.01 * attn_loss)

        # --- Phase 2: GRL Loss ---
        feat_head = self.model.hook_head.output
        if feat_head is not None:
            domain_pred = self.model.grl_head(feat_head)
            dummy_target = torch.zeros(domain_pred.size(0), dtype=torch.long, device=self.device)
            grl_loss = F.cross_entropy(domain_pred, dummy_target)
            aux_loss += (0.05 * grl_loss)

        # 5. Combined Backward Pass
        total_loss = loss + aux_loss
        total_loss.backward()
        
        # 6. Optimizer Step
        self.optimizer.step()
        self.optimizer.zero_grad() 

        # Clear hook outputs to save memory
        self.model.hook_l8.output = None
        self.model.hook_l12.output = None
        self.model.hook_head.output = None

        return loss_items

# --- MAIN ---
if __name__ == '__main__':
    # Force single-GPU to avoid DDP issues
    trainer = MGDATrainer(overrides={'model': MODEL_CFG, 'data': DATA_CFG, 'epochs': EPOCHS, 'device': 0})
    trainer.train()