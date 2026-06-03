#!/usr/bin/env python3
"""
split_reid.py

Копирует файлы из структуры:
  final_reid_dataset/
    20_01/
      0001.jpg
      0002.jpg
    20_02/
      0001.jpg
      ...
в выходную структуру:
  final_reid_dataset_split/
    train_images/
      20_01_0001.jpg
      ...
    test_images/
      20_02_0001.jpg
      ...

По умолчанию берёт каждый N-й файл для test (параметр --step).
--start указывает с какого индекса начинать (0-based). Например,
--step 5 --start 4 => берёт 5-й, 10-й, 15-й...
По умолчанию файлы копируются. Укажите --move для перемеще��ия.
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def natural_key(s: str):
    # Разбивает строку на числа и текст для естественной сортировки
    parts = re.split(r'(\d+)', s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

def gather_image_files(dirpath: Path) -> List[Path]:
    files = [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Split reid dataset into train/test with prefixed filenames.")
    parser.add_argument("--src", required=True, help="Исходная папка (например final_reid_dataset)")
    parser.add_argument("--dst", required=True, help="Папка назначения (например final_reid_dataset_split)")
    parser.add_argument("--step", type=int, default=5, help="Берём каждый N-й файл для test (по умолча��ию 5)")
    parser.add_argument("--start", type=int, default=0, help="С какого индекса начинать (0-based). По умолчанию 0")
    parser.add_argument("--move", action="store_true", help="Перемещать файлы вместо копирования")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists() or not src.is_dir():
        print(f"Ошибка: исходная папка не найдена или не каталог: {src}")
        return

    train_dir = dst / "train_images"
    test_dir = dst / "test_images"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    total = 0
    cnt_train = 0
    cnt_test = 0

    for sub in sorted([p for p in src.iterdir() if p.is_dir()], key=lambda p: p.name):
        whale_id = sub.name
        imgs = gather_image_files(sub)
        if not imgs:
            continue
        for idx, img_path in enumerate(imgs):
            total += 1
            # выбираем каждый step-й с учётом стартовой позиции
            if args.step > 0 and ((idx - args.start) % args.step == 0):
                target_dir = test_dir
                cnt_test += 1
            else:
                target_dir = train_dir
                cnt_train += 1

            new_name = f"{whale_id}_{img_path.name}"
            dst_path = target_dir / new_name

            # Если dst_path существует, добавим суффикс, чтобы не перезаписать
            if dst_path.exists():
                base = dst_path.stem
                ext = dst_path.suffix
                i = 1
                while True:
                    candidate = target_dir / f"{base}_{i}{ext}"
                    if not candidate.exists():
                        dst_path = candidate
                        break
                    i += 1

            if args.move:
                shutil.move(str(img_path), str(dst_path))
            else:
                shutil.copy2(str(img_path), str(dst_path))

    print(f"Готово. Всего файлов обработано: {total}")
    print(f"  train: {cnt_train}")
    print(f"  test : {cnt_test}")
    print(f"Результат в папке: {dst}")

if __name__ == "__main__":
    main()