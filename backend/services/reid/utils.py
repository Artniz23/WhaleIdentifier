import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from typing import Optional

from services.reid.config import Config

def load_df(in_base_dir: str, cfg: Config, filename: str) -> pd.DataFrame:
    df = pd.read_csv(f"{in_base_dir}/{filename}")

    # label encoder
    if hasattr(df, "individual_id"):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.classes_ = np.load(f"{in_base_dir}/individual_id.npy", allow_pickle=True)
        df.individual_id = label_encoder.transform(df.individual_id)
        assert cfg.num_classes == len(label_encoder.classes_)
    return df

def topk_average_precision(output: torch.Tensor, y: torch.Tensor, k: int):
    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)
    topk = output.topk(k)[1]
    match_mat = topk == y[:, None].expand(topk.shape)
    return (match_mat * score_array).sum(dim=1)


def calc_map5(output: torch.Tensor, y: torch.Tensor, threshold: Optional[float]):
    if threshold is not None:
        output = torch.cat([output, torch.full((output.shape[0], 1), threshold, device=output.device)], dim=1)
    return topk_average_precision(output, y, 5).mean().detach()


def map_dict(output: torch.Tensor, y: torch.Tensor, prefix: str):
    d = {f"{prefix}/acc": topk_average_precision(output, y, 1).mean().detach()}
    for threshold in [None, 0.3, 0.4, 0.5, 0.6, 0.7]:
        d[f"{prefix}/map{threshold}"] = calc_map5(output, y, threshold)
    return d