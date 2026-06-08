from pathlib import Path
import argparse
import numpy as np

# Пути по умолчанию для датасета и выходного файла
DEFAULT_DATASET_DIR = "final_reid_dataset"
DEFAULT_OUTPUT_FILE = "individual_id.npy"

def get_whale_ids(dataset_dir: Path) -> list[str]:
    # Идентификаторы китов соответствуют именам директорий
    return sorted(
        d.name
        for d in dataset_dir.iterdir()
        if d.is_dir()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual_id.npy from ReID dataset"
    )

    parser.add_argument(
        "--src",
        default=DEFAULT_DATASET_DIR,
        help="Папка с директориями китов",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Выходной .npy файл",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.src)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    whale_ids = get_whale_ids(dataset_dir)

    # Сохраняем список идентификаторов китов для последующего
    # использования при обучении и инференсе модели
    np.save(
        args.output,
        np.array(whale_ids, dtype=object),
    )

    print(
        f"Сохранено {len(whale_ids)} whale ids "
        f"в {args.output}"
    )

    # Показываем первые несколько идентификаторов для проверки
    print(whale_ids[:10])

if __name__ == "__main__":
    main()