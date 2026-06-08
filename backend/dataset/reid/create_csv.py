"""
Создаёт CSV-файлы train.csv и test.csv из папок:
  <src>/train_images
  <src>/test_images

Формат CSV (заголовок):
image,individual_id

Пример использования:
python3 make_csv_from_split.py --src final_reid_dataset_split
"""
import argparse
import csv
import re
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_DIR = "train_images"
TEST_DIR = "test_images"
CSV_COLUMNS = ["image", "individual_id"]


def natural_key(s: str):
    # Ключ для "естественной" сортировки:
    # image_2.jpg < image_10.jpg
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"([0-9]+)", s)
    ]


def list_images(folder: Path) -> List[Path]:
    # Возвращаем список изображений, отсортированный по имени
    return sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ),
        key=lambda p: natural_key(p.name),
    )


def extract_id_from_filename(fname: str) -> str:
    # 22_04_0008.jpg -> 22_04
    stem = Path(fname).stem
    return "_".join(stem.split("_")[:2])


def make_csv_from_folder(images_folder: Path, out_csv: Path) -> int:
    imgs = list_images(images_folder)
    if not imgs:
        print(f"Warning: нет изображений в {images_folder}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "individual_id"]
        )

        writer.writeheader()

        # Для каждого изображения сохраняем имя файла
        # и идентификатор особи, извлечённый из имени файла
        for p in imgs:
            writer.writerow(
                {
                    "image": p.name,
                    "individual_id": extract_id_from_filename(p.name),
                }
            )
    return len(imgs)


def main():
    parser = argparse.ArgumentParser(
        description="Create train.csv and test.csv from final_reid_dataset_split"
    )

    parser.add_argument(
        "--src",
        default="final_reid_dataset_split",
        help="Папка с train_images и test_images",
    )

    parser.add_argument(
        "--train-name",
        default="train.csv",
        help="Имя выходного CSV для train",
    )

    parser.add_argument(
        "--test-name",
        default="test.csv",
        help="Имя выходного CSV для test",
    )

    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists() or not src.is_dir():
        print(f"Ошибка: папка не найдена или не каталог: {src}")
        return

    # Генерируем CSV для train и test выборок
    datasets = [
        ("train_images", args.train_name),
        ("test_images", args.test_name),
    ]

    for images_dir, csv_name in datasets:
        count = make_csv_from_folder(
            src / images_dir,
            src / csv_name,
        )

        print(
            f"{csv_name} — {count} записей"
        )


if __name__ == "__main__":
    main()
