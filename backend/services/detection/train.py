from ultralytics import YOLO
import argparse
from dotenv import load_dotenv

from core.env import get_required_env


def load_model() -> YOLO:
    """
    Загружает предобученную YOLO OBB модель.
    """
    weights_path = get_required_env(
        "YOLO_PRETRAINED_WEIGHTS_PATH"
    )

    return YOLO(weights_path)


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы обучения.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO OBB model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset.yaml",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Training device.",
    )

    parser.add_argument(
        "--project",
        default="runs_obb",
        help="Output project directory.",
    )

    parser.add_argument(
        "--name",
        default="whale_obb_baseline",
        help="Experiment name.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Rotation augmentation.",
    )

    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability.",
    )

    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Vertical flip probability.",
    )

    return parser.parse_args()


def main():
    """
    Точка входа в обучение YOLO OBB модели.
    """

    load_dotenv()

    args = parse_args()

    model = load_model()

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
    )


if __name__ == "__main__":
    main()
