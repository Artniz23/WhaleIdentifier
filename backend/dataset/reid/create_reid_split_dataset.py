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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Выходные директории для train и test выборок
TRAIN_DIR = "train_images"
TEST_DIR = "test_images"


def natural_key(s: str):
    """Ключ для естественной сортировки файлов."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    ]

def gather_image_files(dirpath: Path) -> list[Path]:
    """Собрать и отсортировать изображения из директории."""
    return sorted(
        (
            p
            for p in dirpath.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ),
        key=lambda p: natural_key(p.name),
    )

def is_test_image(index: int, step: int, start: int) -> bool:
    """
    Определить, должен ли файл попасть в test.

    Пример:
        step=5, start=4

    Тогда в test попадут файлы с индексами:
        4, 9, 14, 19 ...
    """
    return step > 0 and ((index - start) % step == 0)

def get_unique_path(path: Path) -> Path:
    """
    Вернуть уникальный путь.

    Если файл уже существует, добавляет суффикс:
    image.jpg -> image_1.jpg -> image_2.jpg ...
    """
    if not path.exists():
        return path

    base = path.stem
    ext = path.suffix

    i = 1

    while True:
        candidate = path.parent / f"{base}_{i}{ext}"

        if not candidate.exists():
            return candidate

        i += 1

def copy_or_move_file(
    src_path: Path,
    dst_path: Path,
    move: bool,
):
    """Скопировать или переместить файл."""
    if move:
        shutil.move(str(src_path), str(dst_path))
    else:
        shutil.copy2(str(src_path), str(dst_path))

def main():
    parser = argparse.ArgumentParser(
        description="Split ReID dataset into train/test"
    )

    parser.add_argument(
        "--src",
        required=True,
        help="Исходная папка (например final_reid_dataset)",
    )

    parser.add_argument(
        "--dst",
        required=True,
        help="Папка назначения (например final_reid_dataset_split)",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Каждый N-й файл отправлять в test",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Стартовый индекс для отбора в test",
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Перемещать файлы вместо копирования",
    )

    args = parser.parse_args()

    if args.step < 0:
        raise ValueError("--step must be >= 0")

    if args.start < 0:
        raise ValueError("--start must be >= 0")

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"Исходная папка не найдена: {src}"
        )

    train_dir = dst / TRAIN_DIR
    test_dir = dst / TEST_DIR

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    cnt_train = 0
    cnt_test = 0

    # Каждая директория верхнего уровня соответствует одному киту
    whale_dirs = sorted(
        (p for p in src.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )

    for whale_dir in whale_dirs:
        whale_id = whale_dir.name

        images = gather_image_files(whale_dir)

        if not images:
            continue

        for idx, img_path in enumerate(images):
            total += 1

            # Распределяем изображения между train и test
            # согласно шагу (--step) и стартовой позиции (--start)
            if is_test_image(
                    idx,
                    args.step,
                    args.start,
            ):
                target_dir = test_dir
                cnt_test += 1
            else:
                target_dir = train_dir
                cnt_train += 1

            # Добавляем идентификатор кита к имени файла,
            # чтобы сохранить уникальность после объединения
            # всех изображений в одну директорию
            dst_path = get_unique_path(
                target_dir / f"{whale_id}_{img_path.name}"
            )

            copy_or_move_file(
                src_path=img_path,
                dst_path=dst_path,
                move=args.move,
            )

    print(f"Готово. Всего файлов обработано: {total}")
    print(f"  train: {cnt_train}")
    print(f"  test : {cnt_test}")
    print(f"Результат в папке: {dst}")

if __name__ == "__main__":
    main()