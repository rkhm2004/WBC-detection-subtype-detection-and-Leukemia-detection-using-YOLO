import os
from ultralytics import YOLO
import pandas as pd

# ==============================================================================
# 🔴 CONFIGURATION: UPDATE THESE PATHS BEFORE RUNNING
# ==============================================================================
# Path to your standard YOLOv10n 'best.pt' files for each dataset.
# If you haven't trained these specific tasks yet, comment them out.
MODELS = {
    "bccd":     r"D:\PROJECTS\SEM 4\OS\WBC_project\Detection_yolov10n_bccd\Detection_runs6\weights\best.pt", 
    "pbc":      r"D:\PROJECTS\SEM 4\OS\WBC_project\Wbc_subtype_runs_pbc\subtype_yolov10n_pbc\weights\best.pt",
    "ball":     r"D:\PROJECTS\SEM 4\OS\WBC_project\Cancer_Leukemia_V10\v10_leukemia_experiment\weights\best.pt",
    "yolo_new": r"D:\PROJECTS\SEM 4\OS\WBC_project\Baseline_Results_yolo\detect\Baseline_YOLOv10n3\weights\best.pt"
}

# The Dataset YAML files (These should be in your project root)
DATASETS = {
    "bccd":     "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_bccd_datamaster\data_bccd_datamaster.yaml",
    "pbc":      "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_pbc\data_pbc.yaml",
    "ball":     "D:\PROJECTS\SEM 4\OS\WBC_project\dataset\B-ALL\data_B_ALL.yaml",
    "yolo_new": "D:\PROJECTS\SEM 4\OS\WBC_project\yolo_new\data_new.yaml"
}
# ==============================================================================

def main():
    results_list = []
    print("--- STARTING BASELINE (YOLOv10n) TESTING ---")

    for task, weight_path in MODELS.items():
        print(f"\nTesting Baseline on {task.upper()}...")
        
        if not os.path.exists(weight_path):
            print(f"❌ Skipping {task}: Weight file not found at {weight_path}")
            continue

        try:
            # 1. Load Model
            model = YOLO(weight_path)
            
            # 2. Run Test & Save to Specific Folder
            # Results will go to: runs/test_baseline/bccd, runs/test_baseline/pbc, etc.
            metrics = model.val(
                data=DATASETS[task], 
                split='test', 
                project='Test_all_4_data/test_baseline', 
                name=task,
                verbose=False
            )
            
            # 3. Log Results
            results_list.append({
                "Dataset": task.upper(),
                "Model": "Baseline",
                "mAP@50": round(metrics.box.map50 * 100, 2),
                "mAP@50-95": round(metrics.box.map * 100, 2)
            })
            print(f"✅ {task} Done: mAP@50={metrics.box.map50:.4f}")
            
        except Exception as e:
            print(f"❌ Error testing {task}: {e}")

    # 4. Print Summary Table
    print("\n" + "="*50)
    print(" 📊 FINAL BASELINE RESULTS SUMMARY")
    print("="*50)
    if results_list:
        df = pd.DataFrame(results_list)
        print(df.to_string(index=False))
    else:
        print("No models were successfully tested.")

if __name__ == "__main__":
    main()