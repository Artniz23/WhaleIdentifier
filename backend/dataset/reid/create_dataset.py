import cv2
from pathlib import Path


def extract_whale_id(folder_name: str) -> str:
    """
    Из имени папки вида:
      20_01_frames -> 20_01
      20_02_frames -> 20_02

    Если суффикса _frames нет, вернет исходное имя.
    """
    if folder_name.endswith("_frames"):
        return folder_name[:-7]  # убираем "_frames"
    return folder_name


def analyze_whale_dataset(dataset_root):
    """Проанализировать структуру датасета"""
    dataset_root = Path(dataset_root)

    stats = {
        'total_whales': 0,
        'total_images': 0,
        'whales': {}
    }

    whale_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    print("===== АНАЛИЗ ДАТАСЕТА =====\n")

    for whale_dir in whale_dirs:
        whale_folder_name = whale_dir.name          # например 20_01_frames
        whale_id = extract_whale_id(whale_folder_name)  # например 20_01

        images = []
        for subdir in sorted(whale_dir.iterdir()):
            if subdir.is_dir():
                jpg_files = sorted(subdir.glob('*.jpg'))
                images.extend(jpg_files)

        if len(images) == 0:
            print(f"⚠️  {whale_folder_name} ({whale_id}): нет изображений")
            continue

        sizes = []
        for img_path in images[:3]:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w, c = img.shape
                sizes.append((w, h))

        stats['total_whales'] += 1
        stats['total_images'] += len(images)
        stats['whales'][whale_id] = {
            'source_folder': whale_folder_name,
            'count': len(images),
            'sizes': list(set(sizes)),
            'image_paths': [str(p) for p in images]
        }

        print(f"Кит ID: {whale_id}")
        print(f"  Исходная папка: {whale_folder_name}")
        print(f"  Изображений: {len(images)}")
        print(f"  Размеры: {sizes[0] if sizes else 'unknown'}")
        print()

    print(f"ВСЕГО: {stats['total_whales']} китов, {stats['total_images']} изображений")
    if stats['total_whales'] > 0:
        print(f"Среднее на кита: {stats['total_images'] / stats['total_whales']:.1f}")

    return stats


def create_standard_reid_dataset(original_root, output_root):
    """
    Создать стандартизированный датасет

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
    output_root.mkdir(exist_ok=True, parents=True)

    whale_dirs = sorted([d for d in original_root.iterdir() if d.is_dir()])

    print(f"Найдено {len(whale_dirs)} китов\n")

    used_ids = set()

    for original_whale_dir in whale_dirs:
        original_whale_name = original_whale_dir.name   # 20_01_frames
        whale_id = extract_whale_id(original_whale_name)  # 20_01

        if whale_id in used_ids:
            raise ValueError(f"Повторяющийся whale_id: {whale_id}")
        used_ids.add(whale_id)

        new_whale_dir = output_root / whale_id
        new_whale_dir.mkdir(exist_ok=True)

        all_images = []
        for subdir in sorted(original_whale_dir.iterdir()):
            if subdir.is_dir():
                jpg_files = sorted(subdir.glob('*.jpg'))
                all_images.extend(jpg_files)

        print(f"{whale_id} ({original_whale_name}): {len(all_images)} изображений")

        for idx, img_path in enumerate(all_images, 1):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  ❌ Не удалось прочитать: {img_path}")
                    continue

                new_name = f"{idx:04d}.jpg"
                new_path = new_whale_dir / new_name
                cv2.imwrite(str(new_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            except Exception as e:
                print(f"  ❌ Ошибка при обработке {img_path}: {e}")

    print(f"\n✓ Датасет создан в {output_root}")
    print(f"✓ Всего китов: {len(used_ids)}")


stats = analyze_whale_dataset('final_whale_reid_dataset')

create_standard_reid_dataset(
    original_root='final_whale_reid_dataset',
    output_root='final_reid_dataset',
)