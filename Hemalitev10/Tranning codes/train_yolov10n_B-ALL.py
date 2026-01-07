from ultralytics import YOLO

# Move your paths and setup outside or inside the main block
YAML_PATH = r'D:\PROJECTS\SEM 4\OS\WBC_project\dataset\B-ALL\data.yaml'

if __name__ == '__main__':
    # Load YOLOv10n
    model = YOLO('yolov10n.pt') 

    print("\n--- MODEL SUMMARY ---")
    model.info() 
    print("----------------------\n")

    # Start Training
    results = model.train(
        data=YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,      
        project='Cancer_Leukemia_V10',
        name='v10_leukemia_experiment',
        save=True,     
        plots=True     
    )