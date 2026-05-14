import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

def build_eval_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def preprocess_for_eval(image_path, image_size, transform, device):
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    height, width = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(image_size), interpolation=cv2.INTER_CUBIC)

    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    return tensor, width, height

def extract_embedding_tta(model, image_tensor):
    model.eval()

    with torch.no_grad():
        feat1 = model.get_feat(image_tensor)
        feat2 = model.get_feat(torch.flip(image_tensor, dims=[3]))
        feat = (feat1 + feat2) / 2.0
        feat = F.normalize(feat, p=2, dim=1)

    return feat

def get_embedding(model, image_path, device, image_size):
    """
    Возвращает:
    - embedding (np.float32 array)
    - width
    - height
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