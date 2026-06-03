import psycopg2
from pgvector.psycopg2 import register_vector
from tqdm import tqdm
import torch
from pathlib import Path

from services.catalog.utils import get_embedding
from services.reid.classifier import SphereClassifier


def extract_whale_id_from_filename(filename: str) -> str:
    """
    20_01_0001.jpg -> 20_01
    21_17_0337.png -> 21_17
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Неожиданный формат имени файла: {filename}")

    return f"{parts[0]}_{parts[1]}"


def upsert_whale(cur, whale_uid):
    cur.execute("""
        insert into whales (whale_uid)
        values (%s)
        on conflict (whale_uid) do update
        set whale_uid = excluded.whale_uid
        returning id
    """, (whale_uid,))
    return cur.fetchone()[0]


def upsert_catalog_image(cur, whale_id, image_path, width, height):
    cur.execute("""
        insert into catalog_images (
            whale_id, image_path, width, height
        )
        values (%s, %s, %s, %s)
        on conflict (image_path) do update
        set whale_id = excluded.whale_id,
            width = excluded.width,
            height = excluded.height
        returning id
    """, (whale_id, str(image_path), width, height))
    return cur.fetchone()[0]


def upsert_embedding(cur, catalog_image_id, model_name, model_version, embedding):
    cur.execute("""
        insert into image_embeddings (
            catalog_image_id, model_name, model_version, embedding
        )
        values (%s, %s, %s, %s)
        on conflict (catalog_image_id, model_name, model_version) do update
        set embedding = excluded.embedding
    """, (catalog_image_id, model_name, model_version, embedding))


def import_catalog_to_postgres(
        model,
        image_dir,
        device,
        db_config,
        image_size,
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
):
    """
    Импортирует каталог известных китов из плоской папки:
      final_reid_b7_dataset/train_images/
        20_01_0001.jpg
        20_01_0002.jpg
        20_02_0001.jpg
        ...

    whale_uid извлекается из имени файла.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {image_dir}")

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    if len(image_paths) == 0:
        raise ValueError(f"В папке нет изображений: {image_dir}")

    conn = psycopg2.connect(**db_config)
    register_vector(conn)

    total_images = 0
    total_whales = set()

    try:
        with conn:
            with conn.cursor() as cur:
                print(f"=== Импорт каталога из {image_dir} ===")
                print(f"Найдено изображений: {len(image_paths)}")

                for img_path in tqdm(image_paths, desc="Import catalog"):
                    try:
                        whale_uid = extract_whale_id_from_filename(img_path.name)
                        total_whales.add(whale_uid)

                        whale_id = upsert_whale(cur, whale_uid)

                        embedding, width, height = get_embedding(
                            model=model,
                            image_path=img_path,
                            device=device,
                            image_size=image_size,
                        )

                        catalog_image_id = upsert_catalog_image(
                            cur=cur,
                            whale_id=whale_id,
                            image_path=str(img_path),
                            width=width,
                            height=height,
                        )

                        upsert_embedding(
                            cur=cur,
                            catalog_image_id=catalog_image_id,
                            model_name=model_name,
                            model_version=model_version,
                            embedding=embedding,
                        )

                        total_images += 1

                    except Exception as e:
                        print(f"[ERROR] {img_path}: {e}")

        print("\n=== Импорт завершён ===")
        print(f"Уникальных китов: {len(total_whales)}")
        print(f"Изображений обработано: {total_images}")

    finally:
        conn.close()


def main():
    ckpt_path = "models/reid/b7/0/best.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SphereClassifier.load_from_checkpoint(ckpt_path, map_location=device,
                                                  id_class_nums=None,
                                                  backbone_pretrained=False)

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

    import_catalog_to_postgres(
        model=model,
        image_dir="final_reid_b7_dataset/train_images",
        device=device,
        db_config=db_config,
        image_size=(600, 600),
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
    )


if __name__ == "__main__":
    main()
