import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from services.reid.config import Config

class WhaleDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            cfg: Config,
            image_dir: str,
            data_aug: bool,
    ):
        super().__init__()
        self.index = df.index
        self.x_paths = np.array(df.image)
        self.ids = np.array(df.individual_id, dtype=int) if hasattr(df, "individual_id") else np.full(len(df), -1)
        self.cfg = cfg
        self.image_dir = image_dir
        self.df = df
        self.data_aug = data_aug
        augments = []
        if data_aug:
            aug = cfg.aug
            augments = [
                A.Affine(
                    rotate=(-aug.rotate, aug.rotate),
                    translate_percent=(0.0, aug.translate),
                    shear=(-aug.shear, aug.shear),
                    p=aug.p_affine,
                ),
                A.RandomResizedCrop(
                    size=self.cfg.image_size,
                    scale=(aug.crop_scale, 1.0),
                    ratio=(aug.crop_l, aug.crop_r),
                ),
                A.ToGray(p=aug.p_gray),
                A.GaussianBlur(blur_limit=(3, 7), p=aug.p_blur),
                A.GaussNoise(p=aug.p_noise),
                A.Downscale(scale_range=(0.5, 0.5), p=aug.p_downscale),
                A.RandomGridShuffle(grid=(2, 2), p=aug.p_shuffle),
                A.Posterize(p=aug.p_posterize),
                A.RandomBrightnessContrast(p=aug.p_bright_contrast),
                A.CoarseDropout(p=aug.p_cutout),
                A.RandomSnow(p=aug.p_snow),
                A.RandomRain(p=aug.p_rain),
                A.HorizontalFlip(p=0.5),
            ]
        augments.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        augments.append(ToTensorV2())  # HWC to CHW
        self.transform = A.Compose(augments)

    def __len__(self):
        return len(self.ids)

    def get_original_image(self, i: int):
        bgr = cv2.imread(f"{self.image_dir}/{self.x_paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, i: int):
        image = self.get_original_image(i)
        # resize
        image = cv2.resize(image, self.cfg.image_size, interpolation=cv2.INTER_CUBIC)
        # data augmentation
        augmented = self.transform(image=image)["image"]
        return {
            "original_index": self.index[i],
            "image": augmented,
            "label": self.ids[i],
        }