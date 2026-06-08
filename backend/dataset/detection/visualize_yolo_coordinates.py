from pathlib import Path
import cv2
import numpy as np
import argparse

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

FONT_SCALE = 0.7
TEXT_THICKNESS = 2

TEXT_OFFSET = 8
MIN_TEXT_Y = 20

DEFAULT_CLASS_NAMES = {
    0: "Whale",
}

DEFAULT_IMAGES_DIR = "detection_dataset/photos"
DEFAULT_LABELS_DIR = "detection_dataset/labels_obb"
DEFAULT_OUTPUT_DIR = "detection_dataset/visualization_obb"


def read_label_file(label_path: Path) -> list[str]:
    """Прочитать строки из YOLO OBB файла."""

    return [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def parse_obb_line(
        line: str,
) -> tuple[int, list[float]] | None:
    """
    Распарсить строку YOLO OBB.

    Формат:
    class_id x1 y1 x2 y2 x3 y3 x4 y4
    """

    parts = line.split()

    if len(parts) != 9:
        return None

    return (
        int(parts[0]),
        list(map(float, parts[1:])),
    )


def denormalize_points(
        coords: list[float],
        width: int,
        height: int,
) -> np.ndarray:
    """Перевести нормализованные координаты в пиксели."""

    points = [
        [
            int(round(coords[i] * width)),
            int(round(coords[i + 1] * height)),
        ]
        for i in range(0, 8, 2)
    ]

    return np.array(points, dtype=np.int32)


def get_class_label(
        class_id: int,
        class_names: dict[int, str] | None,
) -> str:
    """Получить отображаемое имя класса."""

    if class_names:
        return class_names.get(
            class_id,
            str(class_id),
        )

    return str(class_id)


def draw_obb(
        image: np.ndarray,
        points: np.ndarray,
        label: str,
) -> None:
    """Нарисовать OBB и подпись на изображении."""

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=BOX_COLOR,
        thickness=BOX_THICKNESS,
    )

    x0, y0 = points[0]

    cv2.putText(
        image,
        label,
        (
            x0,
            max(MIN_TEXT_Y, y0 - TEXT_OFFSET),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        BOX_COLOR,
        TEXT_THICKNESS,
    )


def draw_yolo_obb_on_image(
        image_path: Path,
        label_path: Path,
        output_path: Path,
        class_names: dict[int, str] | None = None,
) -> None:
    """Отрисовать YOLO OBB разметку поверх изображения."""

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"[WARN] Не удалось открыть изображение: "
            f"{image_path}"
        )
        return

    height, width = image.shape[:2]

    if not label_path.exists():
        print(
            f"[WARN] Нет файла разметки: "
            f"{label_path}"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(str(output_path), image)
        return

    for line in read_label_file(label_path):
        parsed = parse_obb_line(line)

        if parsed is None:
            print(
                f"[WARN] Некорректная OBB строка: "
                f"{line}"
            )
            continue

        class_id, coords = parsed

        points = denormalize_points(
            coords=coords,
            width=width,
            height=height,
        )

        draw_obb(
            image=image,
            points=points,
            label=get_class_label(
                class_id,
                class_names,
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def visualize_coordinates(
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
) -> None:
    """Создать визуализацию OBB-разметки для всех изображений."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        images_dir.glob("*.jpg")
    )

    for image_path in image_files:
        draw_yolo_obb_on_image(
            image_path=image_path,
            label_path=(
                    labels_dir /
                    f"{image_path.stem}.txt"
            ),
            output_path=(
                    output_dir /
                    image_path.name
            ),
            class_names=DEFAULT_CLASS_NAMES,
        )

        print(
            f"[OK] "
            f"{output_dir / image_path.name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO OBB labels"
    )

    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_DIR,
        help="Папка с изображениями",
    )

    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_DIR,
        help="Папка с YOLO OBB разметкой",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для визуализаций",
    )

    args = parser.parse_args()

    visualize_coordinates(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
