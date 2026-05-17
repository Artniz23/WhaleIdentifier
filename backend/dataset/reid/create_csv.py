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

def natural_key(s: str):
    # Разделяем по числовым группам (без использования \d) для natural sort
    parts = re.split(r'([0-9]+)', s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def extract_id_from_filename(fname: str) -> str:
    """
    Извлекает individual_id из имени файла.
    Ожидается формат: <id>_<digits>.<ext>, например 22_04_0008.jpg -> id=22_04
    """
    # Используем [0-9]+ вместо \d+
    m = re.match(r'^(.+?)_([0-9]+)\.[^.]+$', fname)
    if m:
        return m.group(1)
    if "_" in fname:
        return fname.rsplit(".", 1)[0].split("_")[0]
    return Path(fname).stem

def make_csv_from_folder(images_folder: Path, out_csv: Path) -> int:
    imgs = list_images(images_folder)
    if not imgs:
        print(f"Warning: нет изображений в {images_folder}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "individual_id"])
        count = 0
        for p in imgs:
            name = p.name
            ind = extract_id_from_filename(name)
            writer.writerow([name, ind])
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Create train.csv and test.csv from final_reid_dataset_split")
    parser.add_argument("--src", default="final_reid_dataset_split", help="Папка с train_images и test_images (по умолчанию final_reid_dataset_split)")
    parser.add_argument("--train-name", default="train.csv", help="Имя выходного CSV для train (по умолчанию train.csv)")
    parser.add_argument("--test-name", default="test.csv", help="Имя выходного CSV для test (по умолчанию test.csv)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists() or not src.is_dir():
        print(f"Ошибка: папка не найдена или не каталог: {src}")
        return

    train_dir = src / "train_images"
    test_dir = src / "test_images"
    train_csv = src / args.train_name
    test_csv = src / args.test_name

    n_train = make_csv_from_folder(train_dir, train_csv)
    n_test = make_csv_from_folder(test_dir, test_csv)

    print(f"Созданы:")
    print(f"  {train_csv} — {n_train} записей (включая все строки с данными)")
    print(f"  {test_csv}  — {n_test} записей (включая все строки с данными)")

if __name__ == "__main__":
    main()