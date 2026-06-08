import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from services.reid.config import Config
from services.reid.dataset import WhaleDataset


class WhaleDataModule(LightningDataModule):
    """
    DataModule для подготовки train/validation выборок и создания DataLoader'ов.
    Используется в PyTorch Lightning для организации пайплайна обучения.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            cfg: Config,
            image_dir: str,
            fold: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.image_dir = image_dir
        # Для отладки можно ограничить количество записей датасета.
        if cfg.n_data != -1:
            df = df.iloc[: cfg.n_data]
        # Полный датасет сохраняем отдельно.
        self.all_df = df
        # fold = -1 означает использование всего датасета без валидационного сплита.
        if fold == -1:
            self.train_df = df
        else:
            # StratifiedKFold сохраняет распределение классов между train и val.
            skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=0)
            train_idx, val_idx = list(skf.split(df, df.individual_id))[fold]
            self.train_df = df.iloc[train_idx].copy()
            self.val_df = df.iloc[val_idx].copy()
            # Особи, отсутствующие в train,
            # помечаются как новый (неизвестный) класс.
            new_mask = ~self.val_df.individual_id.isin(self.train_df.individual_id)
            self.val_df.individual_id.mask(new_mask, cfg.num_classes, inplace=True)
            print(f"new: {(self.val_df.individual_id == cfg.num_classes).sum()} / {len(self.val_df)}")

    def get_dataset(self, df, data_aug):
        """
        Создаёт экземпляр WhaleDataset.

        data_aug=True - train аугментации.
        data_aug=False- только preprocessing.
        """
        return WhaleDataset(df, self.cfg, self.image_dir, data_aug)

    def train_dataloader(self):
        """
        DataLoader для обучения.
        Использует аугментации и случайное перемешивание данных.
        """
        dataset = self.get_dataset(self.train_df, True)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        DataLoader для валидации.
        Без аугментаций и без перемешивания.
        """
        if self.cfg.n_splits == -1:
            return None
        return DataLoader(
            self.get_dataset(self.val_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def all_dataloader(self):
        """
        DataLoader для полного датасета.
        Обычно используется для инференса или генерации эмбеддингов.
        """
        return DataLoader(
            self.get_dataset(self.all_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
