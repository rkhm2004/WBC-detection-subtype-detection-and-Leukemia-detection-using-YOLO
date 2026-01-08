import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# --- CONFIG ---
MODELS_DIR = 'Raspberry_models'
MODELS_TO_QUANTIZE = [
    'train_on_B-ALL.onnx',
    'train_on_bccd.onnx',
    'train_on_pbc.onnx',
    'train_on_yolo_new.onnx'
]

def main():
    print(f"🚀 Starting INT8 Quantization for {len(MODELS_TO_QUANTIZE)} models...")
    
    for model_name in MODELS_TO_QUANTIZE:
        input_path = os.path.join(MODELS_DIR, model_name)
        output_name = model_name.replace('.onnx', '_int8.onnx')
        output_path = os.path.join(MODELS_DIR, output_name)
        
        if not os.path.exists(input_path):
            print(f"❌ Skipped (Not Found): {model_name}")
            continue

        print(f"⚡ Quantizing: {model_name} -> {output_name}...")
        
        # This function converts FP32 weights to INT8 dynamically
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
        
        # Check size reduction
        orig_size = os.path.getsize(input_path) / (1024 * 1024)
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Done! Size: {orig_size:.2f} MB -> {new_size:.2f} MB")

    print("\n🎉 All models quantized! Update your test script to use the new '_int8.onnx' files.")

if __name__ == "__main__":
    main()