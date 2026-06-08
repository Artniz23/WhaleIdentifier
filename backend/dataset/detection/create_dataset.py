import shutil
from pathlib import Path
import argparse

from dataset.detection.data_classes import DatasetDirs, DatasetPairs, DatasetSummary

TRAIN_RATIO = 0.8
CLASS_NAME = "Whale"
DEFAULT_FILE_LIST = "detection_dataset/annotations/frames_list.txt"
DEFAULT_PHOTOS_DIR = "detection_dataset/photos"
DEFAULT_LABELS_DIR = "detection_dataset/labels_obb"
DEFAULT_OUTPUT_DIR = "detection_dataset/dataset_obb"
DEFAULT_TRAIN_RATIO = 0.8


def create_dataset_dirs(output_dir: Path) -> DatasetDirs:
    dirs = DatasetDirs(
        train_img=output_dir / "images" / "train",
        val_img=output_dir / "images" / "val",
        train_lbl=output_dir / "labels" / "train",
        val_lbl=output_dir / "labels" / "val",
    )

    for path in vars(dirs).values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_filenames(file_list_path: Path) -> list[str]:
    return [
        line.strip()
        for line in file_list_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def collect_pairs(
        filenames: list[str],
        photos_dir: Path,
        labels_dir: Path,
) -> DatasetPairs:
    pairs = []
    missing_images = []
    missing_labels = []

    for filename in filenames:
        img_path = photos_dir / filename
        lbl_path = labels_dir / f"{Path(filename).stem}.txt"

        if not img_path.exists():
            missing_images.append(filename)
            continue

        if not lbl_path.exists():
            missing_labels.append(lbl_path.name)
            continue

        pairs.append((img_path, lbl_path))

    return DatasetPairs(
        pairs=pairs,
        missing_images=missing_images,
        missing_labels=missing_labels,
    )


def print_missing_files(
        missing_images: list[str],
        missing_labels: list[str],
):
    if missing_images:
        print(
            f"[WARN] Не найдено изображений: "
            f"{len(missing_images)}"
        )

        for name in missing_images[:10]:
            print("   ", name)

    if missing_labels:
        print(
            f"[WARN] Не найдено label-файлов: "
            f"{len(missing_labels)}"
        )

        for name in missing_labels[:10]:
            print("   ", name)


def split_pairs(
        pairs: list[tuple[Path, Path]],
        train_ratio: float,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    n_total = len(pairs)

    n_train = max(
        1,
        int(n_total * train_ratio),
    )

    if n_total > 1:
        n_train = min(
            n_train,
            n_total - 1,
        )

    return (
        pairs[:n_train],
        pairs[n_train:],
    )


def copy_pairs(
        pairs: list[tuple[Path, Path]],
        img_dst: Path,
        lbl_dst: Path,
):
    for img_path, lbl_path in pairs:
        shutil.copy2(
            img_path,
            img_dst / img_path.name,
        )

        shutil.copy2(
            lbl_path,
            lbl_dst / lbl_path.name,
        )


def create_dataset_yaml(
        output_dir: Path,
) -> Path:
    yaml_path = output_dir / "dataset.yaml"

    yaml_path.write_text(
        f"""path: {output_dir.resolve().as_posix()}

train: images/train
val: images/val

names:
  0: {CLASS_NAME}
""",
        encoding="utf-8",
    )

    return yaml_path


def prepare_yolo_dataset(
        file_list_path,
        photos_dir,
        labels_dir,
        output_dir="dataset_obb",
        train_ratio=TRAIN_RATIO,
):
    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1"
        )

    file_list_path = Path(file_list_path)
    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    dirs = create_dataset_dirs(output_dir)

    filenames = load_filenames(file_list_path)

    pairs_info = collect_pairs(
        filenames=filenames,
        photos_dir=photos_dir,
        labels_dir=labels_dir,
    )

    print_missing_files(
        pairs_info.missing_images,
        pairs_info.missing_labels,
    )

    if not pairs_info.pairs:
        raise ValueError(
            "Не найдено ни одной пары image+label"
        )

    train_pairs, val_pairs = split_pairs(
        pairs_info.pairs,
        train_ratio,
    )

    copy_pairs(
        train_pairs,
        dirs.train_img,
        dirs.train_lbl,
    )

    copy_pairs(
        val_pairs,
        dirs.val_img,
        dirs.val_lbl,
    )

    yaml_path = create_dataset_yaml(
        output_dir,
    )

    print(f"[OK] Dataset prepared: {output_dir}")
    print(f"Total used: {len(pairs_info.pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"YAML: {yaml_path}")

    return DatasetSummary(
        output_dir=str(output_dir),
        yaml_path=str(yaml_path),
        total_used=len(pairs_info.pairs),
        train_count=len(train_pairs),
        val_count=len(val_pairs),
        train_first=train_pairs[0][0].name,
        train_last=train_pairs[-1][0].name,
        val_first=(
            val_pairs[0][0].name
            if val_pairs
            else None
        ),
        val_last=(
            val_pairs[-1][0].name
            if val_pairs
            else None
        ),
        missing_images=pairs_info.missing_images,
        missing_labels=pairs_info.missing_labels,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO OBB dataset"
    )

    parser.add_argument(
        "--file-list",
        default=DEFAULT_FILE_LIST,
        help="Файл со списком изображений",
    )

    parser.add_argument(
        "--photos-dir",
        default=DEFAULT_PHOTOS_DIR,
        help="Папка с изображениями",
    )

    parser.add_argument(
        "--labels-dir",
        default=DEFAULT_LABELS_DIR,
        help="Папка с label-файлами",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для результирующего датасета",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Доля train выборки",
    )

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError(
            "--train-ratio must be between 0 and 1"
        )

    prepare_yolo_dataset(
        file_list_path=args.file_list,
        photos_dir=args.photos_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
