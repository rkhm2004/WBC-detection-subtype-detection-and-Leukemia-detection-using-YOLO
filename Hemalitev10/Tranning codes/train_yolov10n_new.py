from ultralytics import YOLO
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov10n.pt")  # lightweight, correct for BCCD

    model.train(
        data=r"D:\PROJECTS\SEM 4\OS\WBC_project\yolo_new\data_new.yaml",
        epochs=50,          # 🔥 DO NOT use 1 epoch
        imgsz=640,
        batch=16,            # safe for RTX 3050 (6GB)
        device=0,           # GPU
        workers=2,
        name="Detection_runs",
        project="Detection_yolov10n_bccd",
        pretrained=True,
        verbose=True
    )

if __name__ == "__main__":
    main()