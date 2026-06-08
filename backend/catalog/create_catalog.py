import psycopg2
from pgvector.psycopg2 import register_vector
from tqdm import tqdm
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
from services.catalog.utils import get_embedding
from services.reid.classifier import SphereClassifier
from catalog.utils import extract_whale_id_from_filename, upsert_whale, upsert_catalog_image, upsert_embedding


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

    # Собираем все поддерживаемые изображения из каталога
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
                        # Получаем идентификатор кита из имени файла:
                        # 20_01_0001.jpg -> 20_01
                        whale_uid = extract_whale_id_from_filename(img_path.name)
                        total_whales.add(whale_uid)

                        # Создаём или обновляем запись о ките
                        whale_id = upsert_whale(cur, whale_uid)

                        # Строим эмбеддинг изображения через ReID-модель
                        embedding, width, height = get_embedding(
                            model=model,
                            image_path=img_path,
                            device=device,
                            image_size=image_size,
                        )

                        # Сохраняем метаданные изображения
                        catalog_image_id = upsert_catalog_image(
                            cur=cur,
                            whale_id=whale_id,
                            image_path=str(img_path),
                            width=width,
                            height=height,
                        )

                        # Сохраняем эмбеддинг с привязкой к версии модели
                        upsert_embedding(
                            cur=cur,
                            catalog_image_id=catalog_image_id,
                            model_name=model_name,
                            model_version=model_version,
                            embedding=embedding,
                        )

                        total_images += 1

                    # Ошибка одного изображения не должна останавливать импорт
                    except Exception as e:
                        print(f"[ERROR] {img_path}: {e}")

        print("\n=== Импорт завершён ===")
        print(f"Уникальных китов: {len(total_whales)}")
        print(f"Изображений обработано: {total_images}")

    finally:
        conn.close()


def main():
    # Загружаем переменные окружения из .env
    load_dotenv()

    # Конфигурация подключения к PostgreSQL
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    # Путь к чекпоинту ReID-модели и каталогу изображений
    CKPT_PATH = os.getenv("REID_CKPT_PATH")
    CATALOG_IMAGE_DIR = os.getenv("CATALOG_IMAGE_DIR")

    # Используем GPU при наличии, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем обученную модель для генерации эмбеддингов
    model = SphereClassifier.load_from_checkpoint(
        CKPT_PATH,
        map_location=device,
        id_class_nums=None,
        backbone_pretrained=False,
    )

    model = model.to(device)
    model.eval()
    model.freeze()

    # Импортируем каталог изображений и сохраняем эмбеддинги в БД
    import_catalog_to_postgres(
        model=model,
        image_dir=CATALOG_IMAGE_DIR,
        device=device,
        db_config=DB_CONFIG,
        image_size=(600, 600),
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
    )


if __name__ == "__main__":
    main()
