import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# --- ⚙️ CONFIGURATION ---
TEST_PAIRS = {
    'train_on_B-ALL.onnx':    'WBC-Malignant-Pre-935_jpg.rf.497e1972cd9bcd0337f7b55921d42a02.jpg',
    'train_on_bccd.onnx':     'BloodImage_00077.jpg',
    'train_on_pbc.onnx':      'BA_45632.jpg',
    'train_on_yolo_new.onnx': 'c7ee4294-2166-4296-8424-ec3bd6c2007b.jpeg'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Raspberry_models')

def preprocess(img, input_shape):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_shape[3], input_shape[2]))
    img_data = img_resized.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    img_data /= 255.0
    return img_data

def main():
    print(f"🚀 Starting Optimized Benchmark (4 Threads / Performance Mode)...\n")
    
    results_table = []

    # --- ⚡ OPTIMIZATION SETTINGS ---
    # These settings are crucial for Raspberry Pi 5
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Force usage of all 4 cores
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # Better for single requests
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3

    for model_file, image_file in TEST_PAIRS.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        image_path = os.path.join(BASE_DIR, image_file)

        if not os.path.exists(model_path):
            results_table.append((model_file, "MISSING MODEL", "-"))
            continue
        if not os.path.exists(image_path):
            results_table.append((model_file, "MISSING IMAGE", "-"))
            continue

        try:
            # Load with optimized settings
            session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
            
            img = cv2.imread(image_path)
            model_inputs = session.get_inputs()
            input_tensor = preprocess(img, model_inputs[0].shape)
            input_name = model_inputs[0].name
        except Exception as e:
            results_table.append((model_file, "ERROR", "-"))
            print(f"Error loading {model_file}: {e}")
            continue

        # Warmup
        session.run(None, {input_name: input_tensor})

        # Measure Speed
        ITERATIONS = 50
        start_time = time.time()
        for _ in range(ITERATIONS):
            session.run(None, {input_name: input_tensor})
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / ITERATIONS) * 1000
        fps = 1000 / avg_time_ms
        
        results_table.append((model_file, f"{avg_time_ms:.2f} ms", f"{fps:.2f} FPS"))

    # --- OUTPUT ---
    print("-" * 65)
    print(f"{'MODEL NAME':<30} | {'INFERENCE TIME':<15} | {'FPS':<10}")
    print("-" * 65)
    
    for row in results_table:
        print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<10}")
            
    print("-" * 65)

if __name__ == "__main__":
    main()