import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# --- ⚙️ CONFIGURATION ---
# Format: { "Model_Filename": "Image_Filename" }
TEST_PAIRS = {
    'train_on_B-ALL_int8.onnx':    'WBC-Malignant-Pre-935_jpg.rf.497e1972cd9bcd0337f7b55921d42a02.jpg',
    'train_on_bccd_int8.onnx':     'BloodImage_00077.jpg',
    'train_on_pbc_int8.onnx':      'BA_45632.jpg',
    'train_on_yolo_new_int8.onnx': 'c7ee4294-2166-4296-8424-ec3bd6c2007b.jpeg'
}

# Settings
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Raspberry_models')

def preprocess(img, input_shape):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_shape[3], input_shape[2]))
    img_data = img_resized.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    img_data /= 255.0
    return img_data

def postprocess(output, img_w, img_h, input_w, input_h):
    outputs = np.transpose(np.squeeze(output[0]))
    boxes, scores, class_ids = [], [], []
    x_scale = img_w / input_w
    y_scale = img_h / input_h

    for row in outputs:
        # The first 4 numbers are the box (x,y,w,h)
        # The rest are class probabilities
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        
        if max_score >= CONF_THRESHOLD:
            class_id = np.argmax(classes_scores)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            left = int((cx - w/2) * x_scale)
            top = int((cy - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            
            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append([boxes[i], scores[i], class_ids[i]])
    return results

def run_test(model_file, image_file):
    model_path = os.path.join(MODELS_DIR, model_file)
    image_path = os.path.join(BASE_DIR, image_file)
    output_path = os.path.join(BASE_DIR, f"result_{model_file[:-5]}.jpg")

    print(f"\n🔹 TESTING PAIR: [{model_file}] + [{image_file}]")

    if not os.path.exists(model_path):
        print(f"   ❌ Model not found: {model_file}")
        return
    if not os.path.exists(image_path):
        print(f"   ❌ Image not found: {image_file}")
        return

    try:
        # Load Model
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        img = cv2.imread(image_path)
        h_orig, w_orig, _ = img.shape
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return

    # Preprocess
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_tensor = preprocess(img, input_shape)

    # Warmup
    session.run(None, {model_inputs[0].name: input_tensor})

    # Speed Test
    ITERATIONS = 20
    t_start = time.time()
    for _ in range(ITERATIONS):
        outputs = session.run(None, {model_inputs[0].name: input_tensor})
    t_total = time.time() - t_start
    
    avg_ms = (t_total / ITERATIONS) * 1000
    fps = 1.0 / (t_total / ITERATIONS)

    # Draw Boxes (Using Generic "Class ID" labels)
    detections = postprocess(outputs, w_orig, h_orig, input_shape[3], input_shape[2])
    
    for box, score, cls_id in detections:
        x, y, w, h = box
        # 🟢 GENERIC LABEL: "Class 0", "Class 1", etc.
        label = f"Class {cls_id} ({score:.2f})"
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)

    print(f"   ✅ Saved: {output_path}")
    print(f"   ⚡ Speed: {avg_ms:.2f} ms ({fps:.2f} FPS)")

if __name__ == "__main__":
    for model, image in TEST_PAIRS.items():
        run_test(model, image)