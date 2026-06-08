import cv2
from pathlib import Path
import argparse

from dataset.reid.data_classes import DatasetStats, WhaleInfo

FRAMES_SUFFIX = "_frames"
# Пути по умолчанию для исходного и результирующего датасетов
DEFAULT_SOURCE_DIR = "final_whale_reid_dataset"
DEFAULT_OUTPUT_DIR = "final_reid_dataset"

def collect_images(whale_dir: Path) -> list[Path]:
    # Собираем все jpg-изображения из вложенных директорий кита
    images = []

    for subdir in sorted(whale_dir.iterdir()):
        if subdir.is_dir():
            images.extend(sorted(subdir.glob("*.jpg")))

    return images

def extract_whale_id(folder_name: str) -> str:
    """
    Из имени папки вида:
    20_01_frames -> 20_01
    20_02_frames -> 20_02

    Если суффикса _frames нет, вернет исходное имя.
    """
    if folder_name.endswith(FRAMES_SUFFIX):
        return folder_name.removesuffix(FRAMES_SUFFIX)
    return folder_name

def get_image_sizes(images: list[Path], sample_size: int = 3):
    # Получаем размеры нескольких изображений для анализа датасета
    sizes = []

    for img_path in images[:sample_size]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w, _ = img.shape
            sizes.append((w, h))

    return sizes

def get_whale_dirs(root: Path) -> list[Path]:
    # Возвращаем список директорий китов
    return sorted(d for d in root.iterdir() if d.is_dir())

def analyze_whale_dataset(dataset_root: str | Path) -> DatasetStats:
    """Проанализировать структуру датасета."""
    dataset_root = Path(dataset_root)

    stats = DatasetStats()

    print("===== АНАЛИЗ ДАТАСЕТА =====\n")

    for whale_dir in get_whale_dirs(dataset_root):
        whale_folder_name = whale_dir.name
        whale_id = extract_whale_id(whale_folder_name)

        images = collect_images(whale_dir)

        if not images:
            print(f"⚠️  {whale_folder_name} ({whale_id}): нет изображений")
            continue

        sizes = get_image_sizes(images, sample_size=3)

        stats.total_whales += 1
        stats.total_images += len(images)

        # Сохраняем статистику по каждому киту
        stats.whales[whale_id] = WhaleInfo(
            source_folder=whale_folder_name,
            count=len(images),
            sizes=list(set(sizes)),
            image_paths=[str(p) for p in images],
        )

        print(f"Кит ID: {whale_id}")
        print(f"  Исходная папка: {whale_folder_name}")
        print(f"  Изображений: {len(images)}")
        print(f"  Размеры: {sizes[0] if sizes else 'unknown'}")
        print()

    print(
        f"ВСЕГО: {stats.total_whales} китов, "
        f"{stats.total_images} изображений"
    )

    if stats.total_whales:
        print(
            f"Среднее на кита: "
            f"{stats.total_images / stats.total_whales:.1f}"
        )

    return stats


def create_standard_reid_dataset(
    original_root: str | Path,
    output_root: str | Path,
):
    """
    Создать стандартизированный датасет.

    Входная структура:
    dataset/
    ├── 20_01_frames/
    │   ├── crop1_DJI_0002/
    │   │   └── crop1_DJI_0002_1.jpg
    │   └── ...
    ├── 20_02_frames/
    └── ...

    Выходная структура:
    reid_dataset/
    ├── 20_01/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    ├── 20_02/
    │   ├── 0001.jpg
    │   └── ...
    """
    original_root = Path(original_root)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    whale_dirs = get_whale_dirs(original_root)

    print(f"Найдено {len(whale_dirs)} китов\n")

    used_ids = set()

    for whale_dir in whale_dirs:
        whale_folder_name = whale_dir.name
        whale_id = extract_whale_id(whale_folder_name)

        # Проверяем, что после удаления суффикса _frames
        # идентификаторы китов остаются уникальными
        if whale_id in used_ids:
            raise ValueError(f"Повторяющийся whale_id: {whale_id}")

        used_ids.add(whale_id)

        target_dir = output_root / whale_id
        target_dir.mkdir(exist_ok=True)

        images = collect_images(whale_dir)

        print(
            f"{whale_id} ({whale_folder_name}): "
            f"{len(images)} изображений"
        )

        # Перенумеровываем изображения внутри каждого кита:
        # 0001.jpg, 0002.jpg, ...
        for idx, img_path in enumerate(images, start=1):
            try:
                img = cv2.imread(str(img_path))

                if img is None:
                    print(f"  ❌ Не удалось прочитать: {img_path}")
                    continue

                target_path = target_dir / f"{idx:04d}.jpg"

                cv2.imwrite(
                    str(target_path),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            # Ошибка одного изображения не должна прерывать обработку датасета
            except Exception as e:
                print(f"  ❌ Ошибка при обработке {img_path}: {e}")

    print(f"\n✓ Датасет создан в {output_root}")
    print(f"✓ Всего китов: {len(used_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare whale ReID dataset"
    )

    parser.add_argument(
        "--src",
        default=DEFAULT_SOURCE_DIR,
        help="Исходный датасет с папками китов",
    )

    parser.add_argument(
        "--dst",
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для результирующего ReID датасета",
    )

    args = parser.parse_args()

    analyze_whale_dataset(args.src)

    create_standard_reid_dataset(
        original_root=args.src,
        output_root=args.dst,
    )


if __name__ == "__main__":
    main()