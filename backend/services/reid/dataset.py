import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from services.reid.config import Config

class WhaleDataset(Dataset):
    """
    Dataset для обучения и инференса модели re-identification китов.

    Отвечает за:
    - чтение изображений;
    - применение аугментаций;
    - нормализацию;
    - преобразование изображений в тензоры.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            cfg: Config,
            image_dir: str,
            data_aug: bool,
    ):
        super().__init__()
        # Индексы исходного DataFrame.
        # Используются для последующего сопоставления результатов.
        self.index = df.index
        # Относительные пути к изображениям.
        self.x_paths = np.array(df.image)
        # Идентификаторы особей (классы).
        # Если столбца нет, используем -1.
        self.ids = np.array(df.individual_id, dtype=int) if hasattr(df, "individual_id") else np.full(len(df), -1)
        self.cfg = cfg
        self.image_dir = image_dir
        self.df = df
        self.data_aug = data_aug
        augments = []
        # Аугментации применяются только во время обучения.
        if data_aug:
            aug = cfg.aug
            augments = [
                # Геометрические преобразования:
                # поворот, сдвиг, наклон.
                A.Affine(
                    rotate=(-aug.rotate, aug.rotate),
                    translate_percent=(0.0, aug.translate),
                    shear=(-aug.shear, aug.shear),
                    p=aug.p_affine,
                ),
                # Случайный кроп и масштабирование.
                A.RandomResizedCrop(
                    size=self.cfg.image_size,
                    scale=(aug.crop_scale, 1.0),
                    ratio=(aug.crop_l, aug.crop_r),
                ),
                # Перевод в оттенки серого.
                A.ToGray(p=aug.p_gray),
                # Размытие изображения.
                A.GaussianBlur(blur_limit=(3, 7), p=aug.p_blur),
                # Добавление случайного шума.
                A.GaussNoise(p=aug.p_noise),
                # Искусственное уменьшение качества изображения.
                A.Downscale(scale_range=(0.5, 0.5), p=aug.p_downscale),
                # Перемешивание частей изображения.
                A.RandomGridShuffle(grid=(2, 2), p=aug.p_shuffle),
                # Уменьшение количества цветов.
                A.Posterize(p=aug.p_posterize),
                # Изменение яркости и контраста.
                A.RandomBrightnessContrast(p=aug.p_bright_contrast),
                # Случайное удаление фрагментов изображения.
                A.CoarseDropout(p=aug.p_cutout),
                # Симуляция снега.
                A.RandomSnow(p=aug.p_snow),
                # Симуляция дождя.
                A.RandomRain(p=aug.p_rain),
                # Горизонтальное отражение.
                A.HorizontalFlip(p=0.5),
            ]
        # Нормализация под статистики ImageNet.
        augments.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        # Преобразование изображения из HWC -> CHW
        # и перевод в torch.Tensor.
        augments.append(ToTensorV2())
        self.transform = A.Compose(augments)

    def __len__(self):
        """
        Возвращает количество объектов в датасете.
        """
        return len(self.ids)

    def get_original_image(self, i: int):
        """
        Загружает исходное изображение и переводит его
        из BGR (OpenCV) в RGB.
        """
        bgr = cv2.imread(f"{self.image_dir}/{self.x_paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, i: int):
        """
        Возвращает один элемент датасета:
        - original_index: индекс в исходном DataFrame;
        - image: подготовленный тензор изображения;
        - label: id особи.
        """
        image = self.get_original_image(i)
        # Приведение изображения к размеру модели.
        image = cv2.resize(image, self.cfg.image_size, interpolation=cv2.INTER_CUBIC)
        # Применение аугментаций и преобразований.
        augmented = self.transform(image=image)["image"]
        return {
            "original_index": self.index[i],
            "image": augmented,
            "label": self.ids[i],
        }