import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# --- Grad-CAM Imports ---
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# =====================================================
# 1. DEFINE CUSTOM MODULES (REQUIRED FOR HEMALITE)
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_planes = max(in_planes // ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

tasks.CBAM = CBAM

# =====================================================
# 2. CONFIGURATION
# =====================================================
TASK_NAME = "pbc"

# 1. IMAGE PATH (Use r"" to avoid path errors)
IMAGE_PATH = r"D:\PROJECTS\SEM 4\OS\WBC_project\yolo_pbc\images\test\EO_390478.jpg"

# 2. MODEL PATH
MODEL_PATH = r"D:\PROJECTS\SEM 4\OS\WBC_project\HemaliteV10_pbc\pbc_run3\weights\best.pt"

# 3. OUTPUT FOLDER
OUTPUT_FOLDER = os.path.join("XAI_Hemalitev10", TASK_NAME)

# =====================================================
# 3. WRAPPER FOR GRAD-CAM (FIXED)
# =====================================================
class YOLOEigenCAMWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        # Access the internal list of layers
        self.model = yolo_model.model.model 
        # This will be populated dynamically in main()
        self.backbone = []

    def forward(self, x):
        # Run through the specific layers we selected
        for layer in self.backbone:
            x = layer(x)
        return x

# =====================================================
# 4. MAIN EXECUTION
# =====================================================
def run_explanation():
    print(f"\n{'='*60}")
    print(f"🕵️‍♂️  STARTING GRAD-CAM (EIGEN-CAM) FOR: {TASK_NAME.upper()}")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    print("   🔄 Loading HemaliteV10 model...")
    yolo_model = YOLO(MODEL_PATH)
    
    # Access the layer list
    base_model_list = yolo_model.model.model 

    # 2. Find Target Layer (SPPF)
    # We look for the SPPF layer to attach Grad-CAM to.
    print("\n   Searching for SPPF layer...")
    target_layer_index = 9 # Default fallback for YOLOv8/v10
    
    for i, layer in enumerate(base_model_list):
        name = layer.__class__.__name__
        if "SPP" in name:
            target_layer_index = i
            print(f"     👉 Found SPPF layer at index [{i}]")
            break
            
    # 3. Setup Wrapper
    wrapper = YOLOEigenCAMWrapper(yolo_model)
    # Important: Slice the model only up to the target layer
    wrapper.backbone = base_model_list[:target_layer_index+1]
    
    wrapper.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(device)

    # 4. Load & Preprocess Image
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        return

    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_float = np.float32(img_resized) / 255.0

    # Standard ImageNet normalization
    input_tensor = preprocess_image(img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(device)

    # 5. Compute EigenCAM
    # Target the very last layer of our sliced backbone
    target_layers = [wrapper.backbone[-1]]

    print(f"\n   Computing EigenCAM heatmap...")
    try:
        cam = EigenCAM(model=wrapper, target_layers=target_layers)
        
        # Generate grayscale heatmap
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        # Overlay on image
        cam_vis = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Stack Original + Heatmap
        combined = np.hstack((img_resized, cam_vis))

        # Save
        img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        save_path = os.path.join(OUTPUT_FOLDER, f"EigenCAM_{img_name}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        print(f"     -> Saved result to: {save_path}")
        print(f"\n✅ GRAD-CAM ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n   ❌ Error computing CAM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_explanation()