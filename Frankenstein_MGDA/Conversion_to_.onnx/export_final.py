import torch
from ultralytics import YOLO

# --- THE BLUEPRINT (Required to load the model in memory) ---
class ActivationHook:
    def __init__(self): self.output = None
    def __call__(self, m, i, o): self.output = o

def export_final_model():
    # 1. Point to your NEW Fine-Tuned Model
    # Note: We are using the 'raabin_finetune' folder now
    model_path = "Frankenstein_Adaptation/raabin_finetune/weights/best.pt"
    
    print(f"🚀 Loading Adapted Frankenstein Model from: {model_path}")
    
    try:
        model = YOLO(model_path)

        # 2. Strip Training Artifacts (Clean it up)
        # Even though they were 'sleeping', they take up file size. We delete them now.
        if hasattr(model.model, 'sr_decoder'): 
            del model.model.sr_decoder
            print("✅ Removed SR Decoder (Gym Equipment)")
            
        if hasattr(model.model, 'grl_head'): 
            del model.model.grl_head
            print("✅ Removed GRL Head (Gym Equipment)")
            
        if hasattr(model.model, 'attn_loss_fn'): 
            del model.model.attn_loss_fn
        
        # Remove any lingering hooks
        for attr in ['hook_l8', 'hook_l12', 'hook_head']:
            if hasattr(model.model, attr): delattr(model.model, attr)

        # 3. Export to ONNX (Jetson Nano Format)
        print("📦 Exporting to ONNX format...")
        # opset=11 is the most compatible for Jetson Nano's TensorRT
        success = model.export(format="onnx", opset=11, simplify=True)
        
        if success:
            print("\n✅ SUCCESS! MISSION ACCOMPLISHED.")
            print(f"Your deployable file is here: {success}")
            print("👉 Transfer 'best.onnx' to your Jetson Nano.")

    except FileNotFoundError:
        print("❌ Error: Could not find the file.")
        print("Check if the folder 'Frankenstein_Adaptation' exists.")

if __name__ == '__main__':
    export_final_model()