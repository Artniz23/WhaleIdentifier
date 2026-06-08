from pathlib import Path

UPSERT_WHALE_QUERY = """
insert into whales (whale_uid)
values (%s)
on conflict (whale_uid) do update
set whale_uid = excluded.whale_uid
returning id
"""

UPSERT_CATALOG_IMAGE_QUERY = """
insert into catalog_images (
    whale_id, image_path, width, height
)
values (%s, %s, %s, %s)
on conflict (image_path) do update
set whale_id = excluded.whale_id,
    width = excluded.width,
    height = excluded.height
returning id
"""

UPSERT_EMBEDDING_QUERY = """
insert into image_embeddings (
    catalog_image_id, model_name, model_version, embedding
)
values (%s, %s, %s, %s)
on conflict (catalog_image_id, model_name, model_version) do update
set embedding = excluded.embedding
    """

def upsert_whale(cur, whale_uid):
    cur.execute(UPSERT_WHALE_QUERY, (whale_uid,))
    return cur.fetchone()[0]


def upsert_catalog_image(cur, whale_id, image_path, width, height):
    cur.execute(UPSERT_CATALOG_IMAGE_QUERY, (whale_id, str(image_path), width, height))
    return cur.fetchone()[0]


def upsert_embedding(cur, catalog_image_id, model_name, model_version, embedding):
    cur.execute(UPSERT_EMBEDDING_QUERY, (catalog_image_id, model_name, model_version, embedding))

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