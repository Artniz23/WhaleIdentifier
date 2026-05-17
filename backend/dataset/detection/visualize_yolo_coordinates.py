from pathlib import Path
import cv2
import numpy as np

def draw_yolo_obb_on_image(image_path, label_path, output_path, class_names=None):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Не удалось открыть изображение: {image_path}")
        return

    h, w = image.shape[:2]

    if not Path(label_path).exists():
        print(f"[WARN] Нет файла разметки: {label_path}")
        cv2.imwrite(str(output_path), image)
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 9:
            print(f"[WARN] Некорректная OBB строка: {line}")
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        pts = []
        for i in range(0, 8, 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)

        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        label = str(class_id)
        if class_names and class_id in class_names:
            label = class_names[class_id]

        x0, y0 = pts[0]
        cv2.putText(
            image,
            label,
            (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

def visualize_coordinates(images_dir, labels_dir, output_dir="visualization_obb"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        output_path = output_dir / image_path.name

        draw_yolo_obb_on_image(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            class_names={0: "Whale"}
        )

        print(f"[OK] {output_path}")

def main():
    visualize_coordinates(
        images_dir="detection_dataset/photos",
        labels_dir="detection_dataset/labels_obb",
        output_dir="detection_dataset/visualization_obb"
    )


if __name__ == "__main__":
    main()
