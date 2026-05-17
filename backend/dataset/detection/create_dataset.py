import shutil
from pathlib import Path

def prepare_yolo_dataset(
        file_list_path,
        photos_dir,
        labels_dir,
        output_dir="dataset_obb",
        train_ratio=0.8
):
    file_list_path = Path(file_list_path)
    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    filenames = [
        line.strip()
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

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

    if missing_images:
        print(f"[WARN] Не найдено изображений: {len(missing_images)}")
        for name in missing_images[:10]:
            print("   ", name)

    if missing_labels:
        print(f"[WARN] Не найдено label-файлов: {len(missing_labels)}")
        for name in missing_labels[:10]:
            print("   ", name)

    if not pairs:
        raise ValueError("Не найдено ни одной пары image+label из списка")

    n_total = len(pairs)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_total

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    def copy_pairs(pairs_list, img_dst, lbl_dst):
        for img_path, lbl_path in pairs_list:
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)

    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: Whale
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"[OK] Dataset prepared: {output_dir}")
    print(f"Total used: {len(pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Train first: {train_pairs[0][0].name}")
    print(f"Train last:  {train_pairs[-1][0].name}")
    if val_pairs:
        print(f"Val first:   {val_pairs[0][0].name}")
        print(f"Val last:    {val_pairs[-1][0].name}")
    print(f"YAML: {yaml_path}")

    return {
        "output_dir": str(output_dir),
        "yaml_path": str(yaml_path),
        "total_used": len(pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_first": train_pairs[0][0].name,
        "train_last": train_pairs[-1][0].name,
        "val_first": val_pairs[0][0].name if val_pairs else None,
        "val_last": val_pairs[-1][0].name if val_pairs else None,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
    }


def main():
    prepare_yolo_dataset(
        file_list_path="detection_dataset/annotations/frames_list.txt",
        photos_dir="detection_dataset/photos",
        labels_dir="detection_dataset/labels_obb",
        output_dir="detection_dataset/dataset_obb",
        train_ratio=0.8
    )


if __name__ == "__main__":
    main()
