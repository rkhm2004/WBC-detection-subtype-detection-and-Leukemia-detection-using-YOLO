import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import pandas as pd

# ==============================================================================
# 1. DEFINE CUSTOM CLASSES (REQUIRED FOR LOADING HEMALITE)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
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
        super(SpatialAttention, self).__init__()
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
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Inject CBAM into Ultralytics
tasks.CBAM = CBAM

# ==============================================================================
# 2. CONFIGURATION: PATHS TO YOUR TRAINED HEMALITE MODELS
# ==============================================================================
# These paths assume you used the 'train_master.py' script I gave you earlier.
# If you saved them elsewhere, please update these lines!
# ==============================================================================
# 🔴 CORRECTED PATHS (Using r"" to fix the backslash error)
# ==============================================================================

# Ensure MODELS paths also use r"" if they are full paths!
MODELS = {
    "bccd":     r"D:\PROJECTS\SEM 4\OS\WBC_project\Hemalitev10_bccd\bccd_run4\weights\best.pt",
    "pbc":      r"D:\PROJECTS\SEM 4\OS\WBC_project\Hemalitev10_pbc\pbc_run3\weights\best.pt",
    "ball":     r"D:\PROJECTS\SEM 4\OS\WBC_project\Hemalitev10_ball\ball_run2\weights\best.pt",
    "yolo_new": r"D:\PROJECTS\SEM 4\OS\WBC_project\Hemalitev10_yolo_new\yolo_new_run\weights\best.pt"
}

# The Dataset YAML files
DATASETS = {
    "bccd":     "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_bccd_datamaster\data_bccd_datamaster.yaml",
    "pbc":      "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_pbc\data_pbc.yaml",
    "ball":     "D:\PROJECTS\SEM 4\OS\WBC_project\dataset\B-ALL\data_B_ALL.yaml",
    "yolo_new": "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_new\data_new.yaml"
}
# ==============================================================================

def main():
    results_list = []
    print("--- STARTING HEMALITEV10 TESTING ---")

    for task, weight_path in MODELS.items():
        print(f"\nTesting HemaliteV10 on {task.upper()}...")
        
        if not os.path.exists(weight_path):
            print(f"❌ Skipping {task}: Weight file not found at {weight_path}")
            continue

        try:
            # 1. Load Model
            model = YOLO(weight_path)
            
            # 2. Run Test & Save to Specific Folder
            # Results will go to: runs/test_hemalite/bccd, runs/test_hemalite/pbc, etc.
            metrics = model.val(
                data=DATASETS[task], 
                split='test', 
                project='Test_Hemalitev10_all_4_data/test_hemalite', 
                name=task,
                verbose=False
            )
            
            # 3. Log Results
            results_list.append({
                "Dataset": task.upper(),
                "Model": "HemaliteV10",
                "mAP@50": round(metrics.box.map50 * 100, 2),
                "mAP@50-95": round(metrics.box.map * 100, 2)
            })
            print(f"✅ {task} Done: mAP@50={metrics.box.map50:.4f}")
            
        except Exception as e:
            print(f"❌ Error testing {task}: {e}")

    # 4. Print Summary Table
    print("\n" + "="*50)
    print(" 📊 FINAL HEMALITE RESULTS SUMMARY")
    print("="*50)
    if results_list:
        df = pd.DataFrame(results_list)
        print(df.to_string(index=False))
    else:
        print("No models were successfully tested.")

if __name__ == "__main__":
    main()