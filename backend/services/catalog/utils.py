import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


def build_eval_transform():
    """
    Создаёт пайплайн преобразований для инференса модели ReID.

    Выполняется только нормализация изображения и преобразование
    в PyTorch Tensor без аугментаций.
    """

    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def preprocess_for_eval(image_path, image_size, transform, device):
    """
    Загружает изображение и подготавливает его для модели.

    Возвращает:
    - tensor для подачи в модель
    - исходную ширину изображения
    - исходную высоту изображения
    """

    image_path = Path(image_path)

    # Читаем изображение с диска.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    # Сохраняем исходный размер изображения.
    height, width = image.shape[:2]

    # OpenCV читает изображения в BGR,
    # а модель ожидает RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Приводим изображение к размеру,
    # использовавшемуся при обучении модели.
    image = cv2.resize(image, tuple(image_size), interpolation=cv2.INTER_CUBIC)

    # Выполняем нормализацию и перенос на устройство.
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    return tensor, width, height


def extract_embedding_tta(model, image_tensor):
    """
    Извлекает эмбеддинг с использованием Test-Time Augmentation (TTA).

    Выполняется два прохода:
    - оригинальное изображение
    - горизонтально отражённое изображение

    После чего признаки усредняются и нормализуются.
    """
    model.eval()

    with torch.no_grad():
        # Эмбеддинг исходного изображения.
        feat1 = model.get_feat(image_tensor)
        # Эмбеддинг зеркально отражённого изображения.
        feat2 = model.get_feat(torch.flip(image_tensor, dims=[3]))
        # Усредняем признаки двух проходов.
        feat = (feat1 + feat2) / 2.0
        # L2-нормализация эмбеддинга.
        feat = F.normalize(feat, p=2, dim=1)

    return feat


def get_embedding(model, image_path, device, image_size):
    """
    Извлекает эмбеддинг изображения для ReID.

    Возвращает:
    - embedding (np.float32)
    - исходную ширину изображения
    - исходную высоту изображения
    """
    transform = build_eval_transform()
    image_tensor, width, height = preprocess_for_eval(
        image_path=image_path,
        image_size=image_size,
        transform=transform,
        device=device,
    )

    emb = extract_embedding_tta(model, image_tensor)
    embedding = emb.cpu().numpy()[0].astype(np.float32)

    return embedding, width, height
