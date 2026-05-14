from services.catalog.identify import identify_whale_group_pgvector
from services.reid.classifier import SphereClassifier
import torch


def main():
    ckpt_path = "models/reid/b7/0/best.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SphereClassifier.load_from_checkpoint(ckpt_path, map_location=device,
                                                  id_class_nums=None)

    model = model.to(device)
    model.eval()
    model.freeze()

    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "whale_reid_db",
        "user": "whale_app",
        "password": "3110"
    }

    group_result = identify_whale_group_pgvector(
        model=model,
        image_paths=[
            "final_reid_b7_dataset/test_images/20_34_0005.jpg",
            "final_reid_b7_dataset/test_images/20_34_0010.jpg",
            "final_reid_b7_dataset/test_images/20_34_0015.jpg",
        ],
        device=device,
        db_config=db_config,
        image_size=(600, 600),
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
        top_k=5,
        search_k=50,
        threshold=1.2,
    )


if __name__ == "__main__":
    main()
