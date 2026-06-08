import os
from typing import List
import pandas as pd
import numpy as np
import argparse
import torch
from pathlib import Path
import scipy.sparse
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

from services.reid.config import Config, load_config


def load_test_images_df(test_dir):
    """
    Загружает список тестовых изображений из директории
    и формирует DataFrame с колонкой image.
    """
    test_dir = Path(test_dir)
    image_files = sorted(test_dir.glob("*.jpg"))
    return pd.DataFrame({"image": [p.name for p in image_files]})


def load_results(results_path: str):
    """
    Загружает результаты инференса из npz-файла
    и восстанавливает исходный порядок объектов
    по original_index.
    """

    results = np.load(results_path)
    # Максимальный индекс + 1 = число объектов.
    n_data = results["original_index"].max() + 1
    # Массив для восстановления порядка.
    ord = np.full(n_data, -1, dtype=int)
    # Для каждого original_index сохраняем позицию
    # соответствующей строки в результирующих массивах.
    ord[results["original_index"]] = np.arange(len(results["original_index"]))
    # Переупорядочиваем все поля согласно original_index.
    ret = {key: results[key][ord] for key in results.files if key != "file_name"}
    # Проверяем, что порядок восстановлен корректно.
    assert np.array_equal(ret["original_index"], np.arange(n_data))
    return ret


def restore_all_pred(n_class: int, pred: np.ndarray, pred_idx: np.ndarray):
    """
    Восстанавливает полную матрицу предсказаний
    размера [N, num_classes] из top-k логитов.
    """
    n_data = pred.shape[0]
    all_pred = np.zeros((n_data, n_class))
    for i in range(n_data):
        all_pred[i][pred_idx[i]] = pred[i]
    return all_pred


def knn_all_pred(n_class: int, n_train: int, test_feat: np.ndarray, train_feat: np.ndarray, train_label: np.ndarray):
    """
    Строит KNN-предсказания по эмбеддингам.

    Для каждого тестового объекта сохраняется
    максимальная cosine similarity для каждого класса.
    """

    neigh = NearestNeighbors(n_neighbors=500, metric="cosine")
    neigh.fit(train_feat)
    test_dist, test_cosine_idx = neigh.kneighbors(test_feat, return_distance=True)  # [n_val, 1000], [n_val, 1000]
    test_cosine = 1 - test_dist
    # Если train_feat содержит объединённые признаки,
    # возвращаем индексы к исходному train набору.
    test_cosine_idx %= n_train
    test_all_knn = np.zeros((len(test_feat), n_class))
    for i, (cosines, idx) in enumerate(zip(test_cosine, test_cosine_idx)):
        pred_ids = train_label[idx]
        for cosine, pred_id in zip(cosines, pred_ids):
            test_all_knn[i][pred_id] = max(test_all_knn[i][pred_id], cosine)
    return test_all_knn


def knn_both_feat(n_class, train_feat1, train_feat2, test_feat1, test_feat2, train_label):
    """
    Выполняет KNN-поиск по двум наборам признаков
    (обычно обычный и TTA flip embedding),
    затем усредняет результаты.
    """

    train_feat12 = np.concatenate([train_feat1, train_feat2], axis=0)
    knn1_both_mat = knn_all_pred(n_class, len(train_label), test_feat1, train_feat12, train_label)
    knn2_both_mat = knn_all_pred(n_class, len(train_label), test_feat2, train_feat12, train_label)
    return (knn1_both_mat + knn2_both_mat) / 2


def binary_search_threshold(n_class: int, mat: torch.Tensor, new_ratio: float) -> float:
    """
    Подбирает порог для класса new_individual так,
    чтобы доля новых объектов приблизительно
    соответствовала new_ratio.
    """

    ok, ng = 0.0, 1.0
    for _ in range(30):
        mid = (ok + ng) / 2
        out_new = torch.cat([mat, torch.full((mat.shape[0], 1), mid, device=mat.device)], dim=1)
        if (out_new.argmax(1) == n_class).to(float).mean() <= new_ratio:
            ok = mid
        else:
            ng = mid
    return ok


def make_submission(
        train_paths: List[str],
        test_paths: List[str],
        args: argparse.Namespace,
        cfg: Config,
        new_ratios: List[float],
        knn_ratio: float = 0.5,
):
    """
    Собирает ансамбль моделей,
    формирует submission и pseudo-label файл.
    """
    os.makedirs("submission", exist_ok=True)
    csr_sum = None
    # Энсемблирование результатов нескольких моделей.
    for train_path, test_path in zip(train_paths, test_paths):
        print(train_path, test_path)
        train_results, test_results = load_results(train_path), load_results(test_path)
        knn_csr = scipy.sparse.csr_matrix(
            knn_both_feat(
                cfg.num_classes,
                train_results["embed_features1"],
                train_results["embed_features2"],
                test_results["embed_features1"],
                test_results["embed_features2"],
                train_results["label"],
            )
        )
        logit_mat_csr = scipy.sparse.csr_matrix(
            restore_all_pred(cfg.num_classes, test_results["pred_logit"], test_results["pred_idx"])
        )
        # Смешиваем классификатор и KNN.
        mat_csr = knn_csr * knn_ratio + logit_mat_csr * (1 - knn_ratio)
        csr_sum = mat_csr if csr_sum is None else csr_sum + mat_csr
    ensembled_mat = (csr_sum / len(train_paths)).todense()
    out = torch.tensor(ensembled_mat)

    # Формируем submission для разных значений new_ratio.
    for new_ratio in new_ratios:
        threshold = binary_search_threshold(cfg.num_classes, out, new_ratio)
        print(f"new_ratio: {new_ratio}, selected threshold: {threshold}")
        out_new = torch.cat([out, torch.full((out.shape[0], 1), threshold, device=out.device)], dim=1)
        top5 = out_new.topk(5)[1]

        # Загружаем отображение class_id - whale_id.
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.classes_ = np.load(f"{args.in_base_dir}/individual_id.npy", allow_pickle=True)
        assert cfg.num_classes == len(label_encoder.classes_)

        def make_str(id_list):
            return " ".join(
                "new_individual" if x == cfg.num_classes else label_encoder.inverse_transform([x])[0] for x in id_list
            )

        df = load_test_images_df(f"{args.in_base_dir}/test_images")
        df["predictions"] = [make_str(id_list) for id_list in top5]
        df.to_csv(f"submission/{args.out_prefix}-{new_ratio}-{threshold}.csv", index=False,
                  columns=["image", "predictions"])

    # Генерируем pseudo-labels для последующего обучения.
    df = load_test_images_df(f"{args.in_base_dir}/test_images")
    top1_conf, top1 = out.max(1)
    df["individual_id"] = label_encoder.inverse_transform(top1)
    df["conf"] = top1_conf
    df.to_csv(
        f"submission/pseudo_label_{args.out_prefix}.csv", index=False, columns=["image", "individual_id", "conf"]
    )


def parse_args():
    """
   Аргументы командной строки.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in-base-dir",
        required=True,
    )

    parser.add_argument(
        "--model-dirs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--out-prefix",
        required=True,
    )

    parser.add_argument(
        "--config-path",
        default="final_reid_config/default.yaml",
    )

    return parser.parse_args()


def main():
    """
    Точка входа для формирования submission.
    """

    args = parse_args()

    cfg = load_config(
        args.config_path,
        args.config_path,
    )

    train_paths = [
        f"{path}/train_results.npz"
        for path in args.model_dirs
    ]

    test_paths = [
        f"{path}/test_results.npz"
        for path in args.model_dirs
    ]

    make_submission(
        train_paths=train_paths,
        test_paths=test_paths,
        args=args,
        cfg=cfg,
        new_ratios=[0.165],
        knn_ratio=0.5,
    )


if __name__ == "__main__":
    main()
