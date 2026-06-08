from core.env import get_required_env
from core.device import get_device
from core.database import build_db_config
from services.catalog.identify import identify_whale_group_pgvector
from services.reid.classifier import SphereClassifier
import torch
import argparse
from dotenv import load_dotenv


def load_model(device: torch.device) -> SphereClassifier:
    """
    Загружает ReID-модель из checkpoint.
    """
    ckpt_path = get_required_env("REID_CKPT_PATH")

    model = SphereClassifier.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        id_class_nums=None,
        backbone_pretrained=False,
    )

    model = model.to(device)
    model.eval()
    model.freeze()

    return model


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Whale group identification using pgvector."
    )

    parser.add_argument(
        "--image-paths",
        nargs="+",
        required=True,
        help="Input image paths.",
    )

    parser.add_argument(
        "--model-name",
        default="EfficientNetB7",
        help="Embedding model name.",
    )

    parser.add_argument(
        "--model-version",
        default="efficientnet_b7_v1",
        help="Embedding model version.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final results.",
    )

    parser.add_argument(
        "--search-k",
        type=int,
        default=50,
        help="Number of candidates retrieved from pgvector.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help="Distance threshold.",
    )

    return parser.parse_args()


def main():
    """
    Точка входа в пайплайн идентификации группы китов.
    """

    load_dotenv()

    args = parse_args()

    device = get_device()
    model = load_model(device)
    db_config = build_db_config()

    result = identify_whale_group_pgvector(
        model=model,
        image_paths=args.image_paths,
        device=device,
        db_config=db_config,
        image_size=(600, 600),
        model_name=args.model_name,
        model_version=args.model_version,
        top_k=args.top_k,
        search_k=args.search_k,
        threshold=args.threshold,
    )

    print(result)


if __name__ == "__main__":
    main()
