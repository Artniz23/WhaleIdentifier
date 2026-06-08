import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from typing import Optional

from services.reid.config import Config

def load_df(in_base_dir: str, cfg: Config, filename: str) -> pd.DataFrame:
    """
    Загружает DataFrame и кодирует individual_id
    в числовые метки классов.
    """
    df = pd.read_csv(f"{in_base_dir}/{filename}")

    # Если в датасете есть идентификаторы особей,
    # преобразуем их в числовые классы через LabelEncoder.
    if hasattr(df, "individual_id"):
        label_encoder = preprocessing.LabelEncoder()
        # Загружаем сохранённый список всех классов.
        label_encoder.classes_ = np.load(f"{in_base_dir}/individual_id.npy", allow_pickle=True)
        # Преобразуем строковые individual_id
        # в числовые индексы классов.
        df.individual_id = label_encoder.transform(df.individual_id)
        # Проверяем, что число классов в конфиге
        # совпадает с количеством классов энкодера.
        assert cfg.num_classes == len(label_encoder.classes_)
    return df

def topk_average_precision(output: torch.Tensor, y: torch.Tensor, k: int):
    """
    Вычисляет Average Precision@K для каждого объекта.

    Если правильный класс находится:
    - на 1 месте -> score = 1.0
    - на 2 месте -> score = 0.5
    - на 3 месте -> score = 0.333
    и т.д.
    """

    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)
    # Индексы top-K наиболее вероятных классов.
    topk = output.topk(k)[1]
    # Матрица совпадений с истинным классом.
    match_mat = topk == y[:, None].expand(topk.shape)
    return (match_mat * score_array).sum(dim=1)


def calc_map5(output: torch.Tensor, y: torch.Tensor, threshold: Optional[float]):
    """
    Вычисляет MAP@5.

    При наличии threshold добавляется дополнительный
    класс "new individual" с фиксированным скором.
    """

    if threshold is not None:
        output = torch.cat([output, torch.full((output.shape[0], 1), threshold, device=output.device)], dim=1)
    return topk_average_precision(output, y, 5).mean().detach()


def map_dict(output: torch.Tensor, y: torch.Tensor, prefix: str):
    """
    Формирует словарь метрик для логирования
    в PyTorch Lightning.
    """

    # Accuracy@1 (правильный класс на первом месте).
    d = {f"{prefix}/acc": topk_average_precision(output, y, 1).mean().detach()}
    # MAP@5 для различных порогов new individual.
    for threshold in [None, 0.3, 0.4, 0.5, 0.6, 0.7]:
        d[f"{prefix}/map{threshold}"] = calc_map5(output, y, threshold)
    return d