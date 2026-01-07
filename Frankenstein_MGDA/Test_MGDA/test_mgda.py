import torch
from ultralytics import YOLO
import multiprocessing

# --- THE BLUEPRINT (Must be at the top level) ---
# This class definition must be visible to all worker processes
class ActivationHook:
    def __init__(self):
        self.output = None
        
    def __call__(self, module, input, output):
        self.output = output

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # This prevents the "spawn" crash on Windows
    multiprocessing.freeze_support() 

    # --- CONFIGURATION ---
    # Double check this path matches your folder structure!
    MODEL_PATH = "runs/detect/train/weights/best.pt" 

    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        print("Starting Evaluation on TEST split...")
        # running validation
        metrics = model.val(data="datasets/pbc_data.yaml", split="test")

        print("\n" + "="*30)
        # Check if metrics are calculated before printing
        if hasattr(metrics.box, 'map50'):
            print(f"✅ Final Test mAP50:    {metrics.box.map50:.4f}")
            print(f"✅ Final Test mAP50-95: {metrics.box.map:.4f}")
        else:
            print("⚠️ Metrics not found. Check if the test set has labels.")
        print("="*30)

    except FileNotFoundError:
        print("❌ Error: Could not find the model file.")
        print(f"Please check if '{MODEL_PATH}' is the correct folder.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")