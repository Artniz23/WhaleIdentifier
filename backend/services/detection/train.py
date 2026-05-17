from ultralytics import YOLO

def main():
    model = YOLO("yolo11s-obb.pt")

    model.train(
        data="drive/MyDrive/dataset_obb/dataset.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,
        device=0,
        project="runs_obb",
        name="whale_obb_baseline",
        patience=15,
        workers=4,
        degrees=0.0,
        fliplr=0.5,
        flipud=0.0
    )

if __name__ == "__main__":
    main()