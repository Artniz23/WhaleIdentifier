import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from pathlib import Path
from collections import defaultdict

from services.catalog.utils import get_embedding


def aggregate_multi_image_predictions(
        per_image_results,
        top_k=5,
        threshold=1.2,
        whale_key="whale_id",
):
    """
    Агрегирует результаты по нескольким изображениям одного кита.

    per_image_results: список dict-ов, где каждый dict содержит:
      {
        "status": "known" | "new_whale",
        "best_distance": float,
        "top_k": [
            {
                whale_key: str,
                "distance": float,
                "best_match_image": str
            },
            ...
        ]
      }

    whale_key:
      - "whale_id" для FAISS
      - "whale_uid" для pgvector

    Логика:
    - суммируем score = 1/(distance + eps) по киту
    - aggregated_distance = median(best_distance по кадрам)
    - если aggregated_distance > threshold => new_whale
    """
    whale_scores = defaultdict(float)
    whale_best_distance = {}
    whale_best_image = {}
    best_distances = []

    for result in per_image_results:
        if "best_distance" in result and result["best_distance"] is not None:
            best_distances.append(float(result["best_distance"]))

        for item in result.get("top_k", []):
            whale = item[whale_key]
            dist = float(item["distance"])
            ref_image = item["best_match_image"]

            score = 1.0 / (dist + 1e-6)
            whale_scores[whale] += score

            if whale not in whale_best_distance or dist < whale_best_distance[whale]:
                whale_best_distance[whale] = dist
                whale_best_image[whale] = ref_image

    if len(whale_scores) == 0:
        return {
            "status": "new_whale",
            "aggregated_distance": None,
            "top_k": [],
            "per_image_results": per_image_results,
        }

    sorted_whales = sorted(whale_scores.items(), key=lambda x: x[1], reverse=True)

    aggregated_distance = float(np.median(best_distances)) if len(best_distances) > 0 else None
    best_whale, best_score = sorted_whales[0]

    aggregated_top_k = []
    for whale, score in sorted_whales[:top_k]:
        aggregated_top_k.append({
            whale_key: whale,
            "score": float(score),
            "best_distance": float(whale_best_distance[whale]),
            "best_match_image": whale_best_image[whale],
        })

    if aggregated_distance is None or aggregated_distance > threshold:
        return {
            "status": "new_whale",
            "aggregated_distance": aggregated_distance,
            "nearest_known_whale": best_whale,
            "nearest_known_image": whale_best_image.get(best_whale),
            "top_k": aggregated_top_k,
            "per_image_results": per_image_results,
        }

    return {
        "status": "known",
        "best_whale": best_whale,
        "aggregated_distance": aggregated_distance,
        "top_k": aggregated_top_k,
        "per_image_results": per_image_results,
    }

def identify_whale_pgvector(
        model,
        image_path,
        device,
        db_config,
        image_size,
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
        top_k=5,
        search_k=50,
        threshold=1.2,
):
    """
    Идентификация query-изображения по pgvector.

    threshold здесь для L2-distance по L2-нормированным эмбеддингам.
    Его нужно калибровать отдельно на validation.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    query_embedding, width, height = get_embedding(
        model=model,
        image_path=image_path,
        device=device,
        image_size=image_size,
    )

    sql = """
    with candidates as (
        select
            w.id as whale_id,
            w.whale_uid,
            ci.id as catalog_image_id,
            ci.image_path,
            ie.embedding <=> %s::vector as distance
        from image_embeddings ie
        join catalog_images ci on ci.id = ie.catalog_image_id
        join whales w on w.id = ci.whale_id
        where ie.model_name = %s
          and ie.model_version = %s
        order by ie.embedding <=> %s::vector
        limit %s
    ),
    best_per_whale as (
        select distinct on (whale_id)
            whale_id,
            whale_uid,
            catalog_image_id,
            image_path,
            distance
        from candidates
        order by whale_id, distance asc
    )
    select
        whale_id,
        whale_uid,
        catalog_image_id,
        image_path,
        distance
    from best_per_whale
    order by distance asc
    limit %s;
    """

    conn = psycopg2.connect(**db_config)
    register_vector(conn)

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_embedding,
                    model_name,
                    model_version,
                    query_embedding,
                    search_k,
                    top_k,
                )
            )
            rows = cur.fetchall()

        if not rows:
            raise ValueError("Поиск не вернул кандидатов")

        best_whale_id, best_whale_uid, _, best_image_path, best_distance = rows[0]

        if best_distance > threshold:
            return {
                "status": "new_whale",
                "best_distance": float(best_distance),
                "nearest_known_whale": best_whale_uid,
                "nearest_known_image": best_image_path,
            }

        top_results = []
        for whale_id, whale_uid, catalog_image_id, ref_image_path, distance in rows:
            top_results.append({
                "whale_id": whale_id,
                "whale_uid": whale_uid,
                "distance": float(distance),
                "best_match_image": ref_image_path,
            })

        return {
            "status": "known",
            "best_whale_id": best_whale_id,
            "best_whale_uid": best_whale_uid,
            "best_distance": float(best_distance),
            "top_k": top_results,
        }

    finally:
        conn.close()

def identify_whale_group_pgvector(
        model,
        image_paths,
        device,
        db_config,
        image_size,
        model_name="EfficientNetB7",
        model_version="efficientnet_b7_v1",
        top_k=5,
        search_k=50,
        threshold=1.2,
):
    per_image_results = []

    for image_path in image_paths:
        result = identify_whale_pgvector(
            model=model,
            image_path=image_path,
            device=device,
            db_config=db_config,
            image_size=image_size,
            model_name=model_name,
            model_version=model_version,
            top_k=top_k,
            search_k=search_k,
            threshold=threshold,
        )
        result["query_image"] = str(image_path)
        per_image_results.append(result)

    aggregated = aggregate_multi_image_predictions(
        per_image_results=per_image_results,
        top_k=top_k,
        threshold=threshold,
        whale_key="whale_uid",
    )

    return aggregated