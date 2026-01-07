import torch
from ultralytics import YOLO
import multiprocessing

# --- THE BLUEPRINT (Required for loading the Frankenstein Model) ---
class ActivationHook:
    def __init__(self):
        self.output = None  
    def __call__(self, module, input, output):
        self.output = output

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 1. Load your PBC-Trained Model (The "Teacher")
    MODEL_PATH = "runs/detect/train/weights/best.pt"  # <--- CHECK THIS PATH
    
    # 2. Point to the NEW Dataset (The "Alien Exam")
    # This dataset has different lighting/cameras than what the model saw during training.
    NEW_DATASET = "datasets/raabin_data.yaml"

    try:
        print(f"🚀 Loading Universal Model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        print(f"🌍 Starting Domain Generalization Test on: {NEW_DATASET}")
        print("    (If accuracy is >60%, MGDA is working correctly)")
        
        # We run validation on the 'test' split of Raabin
        metrics = model.val(data=NEW_DATASET, split="test")

        print("\n" + "="*40)
        print(f"✅ UNIVERSAL SCORE (mAP50):    {metrics.box.map50:.4f}")
        print("="*40)
        
        print("\nInterpretation:")
        print("- 80-90%: Incredible. Your MGDA framework completely solved the domain shift.")
        print("- 50-70%: Good. The model generalizes, but the domain shift is heavy.")
        print("- < 40%:  Failure. The domain shift (lighting/microscope) confused the model.")

    except Exception as e:
        print(f"❌ Error: {e}")