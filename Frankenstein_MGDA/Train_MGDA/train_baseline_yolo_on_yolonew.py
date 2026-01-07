from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the Standard Model (Official Weights)
    model = YOLO('yolo11n.pt') 

    # 2. Train on the New Dataset
    model.train(
        data="datasets/base_paper.yaml",  # Points to your friend's data
        epochs=50,
        imgsz=640,
        project="Base_Paper_Experiment",
        name="baseline_yolo_on_yolonew"
    )